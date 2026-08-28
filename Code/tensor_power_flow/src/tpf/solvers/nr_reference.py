import numpy as np
import pandapower as pp
import time

from tpf.core.results import PowerFlowResult
from tpf.solvers.base_solver import BaseSolver
from tpf.core.network import NetworkData
from numpy.typing import NDArray


class PandapowerNRSolver(BaseSolver):
    """Wrapper um pandapower Newton-Raphson als Referenzlösung (mit PV-Support)."""

    def __init__(self, tol: float = 1e-8, max_iter: int = 100):
        super().__init__(tol, max_iter)

    def solve(self, network: NetworkData) -> PowerFlowResult:
        raise NotImplementedError(
            "Verwende solve_from_net(net) mit einem pandapower-Netz."
        )

    def solve_batch(self, network: NetworkData, s_batch: NDArray) -> PowerFlowResult:
        raise NotImplementedError(
            "Batch-Lösung über pandapower nicht unterstützt."
        )

    def solve_from_net(self, net: pp.pandapowerNet) -> PowerFlowResult:
        """
        Löst den Lastfluss auf einem pandapower-Netz.
        Extrahiert zusätzlich PV-Knoten-Ergebnisse (Q, V_setpoint).
        """
        t_start = time.perf_counter()

        pp.runpp(
            net,
            algorithm="nr",
            tolerance_mva=self.tol,
            max_iteration=self.max_iter,
            enforce_q_lims=False,
        )

        # Slack-Leistung aus pandapower auslesen
        base_mva = net._ppc["baseMVA"]
        p_slack_nr = net.res_ext_grid["p_mw"].values / base_mva
        q_slack_nr = net.res_ext_grid["q_mvar"].values / base_mva
        s_slack_nr = (p_slack_nr + 1j * q_slack_nr).reshape(-1, 1)

        elapsed = time.perf_counter() - t_start

        # --- Spannungen aller Busse ---
        vm_pu = net.res_bus["vm_pu"].values
        va_deg = net.res_bus["va_degree"].values
        v_complex = vm_pu * np.exp(1j * np.deg2rad(va_deg))

        # --- Iterationen ---
        iterations = -1
        if hasattr(net, "_ppc") and net._ppc is not None:
            iterations = net._ppc.get("iterations", -1)

        # --- PV-Knoten-Ergebnisse extrahieren ---
        pv_indices, pv_q_pu, pv_v_setpoint = self._extract_pv_results(net)

        return PowerFlowResult(
            voltages=v_complex,
            iterations=iterations,
            converged=net.converged,
            elapsed_time_s=elapsed,
            max_mismatch=0.0,
            s_slack=s_slack_nr,
            pv_indices=pv_indices,
            pv_q_pu=pv_q_pu,
            pv_v_setpoint_pu=pv_v_setpoint,
        )

    def _extract_pv_results(self, net: pp.pandapowerNet):
        """
        Extrahiert PV-Knoten-Daten aus dem gelösten pandapower-Netz.

        Returns
        -------
        pv_indices : ndarray (PPC-Busindizes der PV-Knoten)
        pv_q_pu : ndarray (Blindleistung in p.u. an jedem PV-Knoten)
        pv_v_setpoint : ndarray (Sollspannungsbetrag in p.u.)
        """
        ppc = net._ppc
        bus_types = ppc["bus"][:, 1].astype(int)
        pv_idx = np.where(bus_types == 2)[0]

        if len(pv_idx) == 0:
            return None, None, None

        base_mva = ppc["baseMVA"]

        # Sollspannung der PV-Knoten (aus PPC bus-Daten, Spalte 7 = Vm)
        pv_v_setpoint = ppc["bus"][pv_idx, 7]  # p.u.

        # Q an PV-Knoten: Summe aus gen-Einspeisung - Last
        # Generatoren am PV-Bus
        gen_buses = ppc["gen"][:, 0].astype(int)
        pv_q_pu = np.zeros(len(pv_idx))

        for i, bus in enumerate(pv_idx):
            # Generator-Q an diesem Bus (in MW → p.u.)
            gen_mask = gen_buses == bus
            q_gen = np.sum(ppc["gen"][gen_mask, 2]) / base_mva  # Qg in p.u.

            # Last-Q an diesem Bus
            q_load = ppc["bus"][bus, 3] / base_mva  # Qd in p.u.

            # Netto-Einspeisung (aus Sicht der Netzgleichung)
            pv_q_pu[i] = q_gen - q_load

        return pv_idx, pv_q_pu, pv_v_setpoint

    def measure_overhead_ms(self, net: pp.pandapowerNet, repeats: int = 5) -> float:
        """
        Schätzt den konstanten pandapower-Aufrufoverhead: runpp auf einem
        bereits gelösten Netz mit init='results' braucht ~1 NR-Iteration,
        die gemessene Zeit ist also ≈ Overhead + 1 Iteration.
        Damit lässt sich t_NR,solve ≈ t_NR,full − t_overhead abschätzen.
        """
        import copy
        w = copy.deepcopy(net)
        pp.runpp(w, algorithm="nr", tolerance_mva=self.tol,
                 max_iteration=self.max_iter, enforce_q_lims=False)
        ts = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            pp.runpp(w, algorithm="nr", init="results", tolerance_mva=self.tol,
                     max_iteration=self.max_iter, enforce_q_lims=False)
            ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts))


    def solve_timeseries(
        self,
        net: pp.pandapowerNet,
        pq_p_profile_mw: NDArray,
        pq_q_profile_mvar: NDArray,
        pv_p_profile_mw: NDArray | None = None,
        verbose: bool = False,
        collect_stats: bool = True,
    ) -> PowerFlowResult:
        """
        Sequentielle runpp-Schleife. Legt Diagnostik auf der Instanz ab:
          last_iters_per_scenario, last_converged_per_scenario,
          last_solve_time_only_s  (nur runpp-Aufrufe, ohne Profilzuweisung)
        """
        import copy
        t_start = time.perf_counter()

        if not hasattr(net, "_ppc") or net._ppc is None:
            pp.runpp(net, algorithm="nr")
        base_mva = float(net._ppc["baseMVA"])

        tau = pq_p_profile_mw.shape[1]
        n_bus = len(net.bus)
        V_all = np.zeros((n_bus, tau), dtype=np.complex128)
        s_slack_all = np.zeros((1, tau), dtype=np.complex128)
        conv_ps = np.zeros(tau, dtype=bool)
        iters_ps = np.zeros(tau, dtype=np.int32)

        p_gen_base = net.gen["p_mw"].values.copy() if len(net.gen) > 0 else None
        net_work = copy.deepcopy(net)
        t_solve_only = 0.0

        for t in range(tau):
            net_work.load.loc[:, "p_mw"] = pq_p_profile_mw[:, t]
            net_work.load.loc[:, "q_mvar"] = pq_q_profile_mvar[:, t]
            if pv_p_profile_mw is not None and p_gen_base is not None:
                net_work.gen.loc[:, "p_mw"] = pv_p_profile_mw[:, t]

            t0 = time.perf_counter()
            try:
                pp.runpp(net_work, algorithm="nr", tolerance_mva=self.tol,
                         max_iteration=self.max_iter, enforce_q_lims=False)
                t_solve_only += time.perf_counter() - t0
                conv_ps[t] = bool(net_work.converged)
                iters_ps[t] = net_work._ppc.get("iterations", -1)
                vm = net_work.res_bus["vm_pu"].values
                va = net_work.res_bus["va_degree"].values
                V_all[:, t] = vm * np.exp(1j * np.deg2rad(va))
                s_slack_all[0, t] = (
                    net_work.res_ext_grid["p_mw"].values[0] / base_mva
                    + 1j * net_work.res_ext_grid["q_mvar"].values[0] / base_mva
                )
            except Exception:
                t_solve_only += time.perf_counter() - t0
                conv_ps[t] = False
                iters_ps[t] = -1

            if verbose and (t + 1) % max(1, tau // 20) == 0:
                print(f"  NR {t+1}/{tau} — {100*(t+1)/tau:.0f}%")

        elapsed = time.perf_counter() - t_start

        if collect_stats:
            self.last_iters_per_scenario = iters_ps
            self.last_converged_per_scenario = conv_ps
            self.last_solve_time_only_s = t_solve_only

        return PowerFlowResult(
            voltages=V_all,
            iterations=int(np.max(iters_ps)) if tau > 0 else 0,
            converged=bool(np.all(conv_ps)),
            elapsed_time_s=elapsed,
            max_mismatch=0.0,
            s_slack=s_slack_all,
        )