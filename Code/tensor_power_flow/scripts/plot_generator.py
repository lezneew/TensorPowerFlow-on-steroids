USE_PGF = False

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
if USE_PGF:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SAVE_DIR = Path(r"C:\Users\sgrigorevski-admin\TensorPowerFlow\TensorPowerFlow-on-steroids\Bachelor_tensorflow\figures")

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.tpf_dense import TPFDenseSolver
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.network_generator_salazar import get_salazar_scaling_networks, get_salazar_low_rx10_networks, get_salazar_pq_size_sweep
import pandapower as pp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from convergence_analysis import (
    compute_spectral_radius_diagonal,
    compute_spectral_radius_corrected,
    compute_empirical_contraction,
)
from validate_pv_method_a_comprehensive import run_validation_suite


def plot_baseline_tpf_vs_nr(save_name="baseline_tpf_vs_nr.pgf"):
    """
    Baseline-Performance: TPF (Dense) vs. NR for PQ-only networks.
    Replicates benchmarks from Salazar et al. (2024), Fig. 5(a).
    Generates log-log plot of computing time vs network size and speedup plot.

    Parameters
    ----------
    save_name : str
        Output filename for plot
    """
    networks = get_salazar_scaling_networks()
    # Filter to only include PQ-only networks (n_pv=0)
    networks = {k: v for k, v in networks.items() if v["n_pv"] == 0}
    n_repeats = 5

    print(f"\n  Running TPF vs NR baseline on {len(networks)} networks...")

    records = []
    for name, info in networks.items():
        print(f"  Benchmarking {name}...", end=" ")
        record = {"name": name, "error": None}

        try:
            net = info["constructor"]()

            nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
            nr_result = nr_solver.solve_from_net(net)
            if not nr_result.converged:
                record["error"] = "NR divergiert"
                print(f"ERROR: {record['error']}")
                records.append(record)
                continue

            nr_times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                pp.runpp(net, algorithm="nr", tolerance_mva=1e-8, max_iteration=100)
                nr_times.append((time.perf_counter() - t0) * 1000)

            record["nr_converged"] = True
            record["nr_iter"] = nr_result.iterations
            record["nr_time_ms"] = float(np.median(nr_times))

            network = build_network_from_pandapower(net, include_pv=False)
            record["n_bus"] = network.n_bus_phases

            Z_B = np.linalg.inv(network.Y_dd)
            scaling = np.conj(network.s_nom)
            M = Z_B * scaling.reshape(1, -1)
            record["eta"] = float(np.max(np.sum(np.abs(M), axis=0)))

            tpf_solver = TPFDenseSolver(tol=1e-6, max_iter=100)
            tpf_result = tpf_solver.solve(network)

            tpf_times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                tpf_result = tpf_solver.solve(network)
                tpf_times.append((time.perf_counter() - t0) * 1000)

            record["tpf_converged"] = tpf_result.converged
            record["tpf_iter"] = tpf_result.iterations
            record["tpf_time_ms"] = float(np.median(tpf_times))

            ppc = net._ppc
            bus_types = ppc["bus"][:, 1].astype(int)
            pq_idx = np.where(bus_types == 1)[0]

            v_tpf = tpf_result.voltages.flatten()
            v_nr = nr_result.voltages[pq_idx]
            record["max_v_error"] = float(np.max(np.abs(np.abs(v_tpf) - np.abs(v_nr))))

            if record["tpf_time_ms"] > 0:
                record["speedup"] = record["nr_time_ms"] / record["tpf_time_ms"]
            else:
                record["speedup"] = 0.0

            print(f"n={record['n_bus']}, NR={record['nr_time_ms']:.2f}ms, TPF={record['tpf_time_ms']:.2f}ms, Speedup={record['speedup']:.1f}x")

        except Exception as e:
            record["error"] = str(e)
            try:
                print(f"ERROR: {e}")
            except UnicodeEncodeError:
                error_ascii = record['error'].encode('ascii', errors='replace').decode('ascii')
                print(f"ERROR: {error_ascii}")

        records.append(record)

    valid = [r for r in records if r.get("tpf_converged") and r.get("nr_converged") and not r.get("error")]
    valid = sorted(valid, key=lambda r: r["n_bus"])

    print(f"\n  Generating LaTeX table...")

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{r|r|r|r|r|r|r}")
    print(r"\hline")
    print(r"$n_{bus}$ & $\eta$ & NR Iter & NR Zeit (ms) & TPF Iter & TPF Zeit (ms) & Speedup \\")
    print(r"\hline")

    for r in valid:
        eta_str = f"{r['eta']:.3f}" if r['eta'] < 100 else f"{r['eta']:.1f}"
        print(f"{r['n_bus']} & {eta_str} & {r['nr_iter']} & {r['nr_time_ms']:.2f} & {r['tpf_iter']} & {r['tpf_time_ms']:.2f} & {r['speedup']:.2f}x\\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{TPF vs. NR: PQ-Netze (Salazar Skalierung)}")
    print(r"\label{tab:tpf_vs_nr_baseline}")
    print(r"\end{table}")

    if valid:
        speedups = [r["speedup"] for r in valid if r["speedup"] > 0]
        print(f"\n  Speedup TPF/NR: min={min(speedups):.2f}x, max={max(speedups):.2f}x, median={np.median(speedups):.2f}x")

    print(f"\n  Generating plot...")

    n_bus = np.array([r["n_bus"] for r in valid])
    nr_t = np.array([r["nr_time_ms"] for r in valid])
    tpf_t = np.array([r["tpf_time_ms"] for r in valid])
    speedups = nr_t / tpf_t

    fig, ax1 = plt.subplots(figsize=(5.91, 3.5))

    ax1.loglog(n_bus, nr_t, "s--", color="black", markersize=6, linewidth=1.5,
               label="NR", alpha=0.8)
    ax1.loglog(n_bus, tpf_t, "o-", color="black", markersize=6, linewidth=1.5,
               label="TPF", alpha=0.8)

    ax1.set_xlabel(r"$n_{\mathrm{bus}}$", fontsize=12)
    ax1.set_ylabel("Zeit (ms)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)



    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_outer_convergence_error_decoupled(
    network_names: list = None,
    save_name: str = "outer_convergence_error_decoupled.pgf"
):
    """
    Plot max(|V| - V_spec) vs outer iteration using decoupled (diagonal) Q-update.
    Formula: ΔQ_k = (|V_spec|² - |v|²) / (2·X_kk)

    Parameters
    ----------
    network_names : list
        List of network names to test. Default: ["sz_40_r010", "sz_40_r020"]
    save_name : str
        Output filename
    """
    if network_names is None:
        network_names = ["sz_40_r010", "sz_40_r020"]

    networks = get_salazar_scaling_networks()
    fig, ax = plt.subplots(figsize=(5.91, 3.5))

    colors = ["darkblue", "darkred"]

    for idx, name in enumerate(network_names):
        print(f"  Processing: {name}...")
        info = networks[name]
        net_pandapower = info["constructor"]()

        network = build_network_from_pandapower(net_pandapower, include_pv=True)

        if not network.has_pv or network.n_pv == 0:
            print(f"    Warning: {name} has no PV nodes, skipping")
            continue

        solver = TPFDensePVMethodA(
            tol=1e-8,
            max_iter_inner=100,
            max_iter_outer=100,
            tol_pv=1e-6,
            omega=1.0,
            use_decoupled=True
        )

        result = solver.solve(network)

        if solver.pv_info is None or not hasattr(solver.pv_info, 'pv_v_error_history'):
            print(f"    Warning: No PV error history for {name}")
            continue

        pv_errors = solver.pv_info.pv_v_error_history
        if not pv_errors:
            print(f"    Warning: Empty PV error history for {name}")
            continue
        linestyle = "x--" if not result.converged else "o--"
        iterations = np.arange(1, len(pv_errors) + 1)
        label = f"$n_{{\mathrm{{pv}}}}={network.n_pv}$"
        ax.loglog(iterations, pv_errors, linestyle, color=colors[idx],
                    linewidth=1.5, markersize=6, label=label, alpha=0.8)

    # ax.axhline(y=1e-6, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label=r"$10^{-6}$ tolerance")
    ax.set_xlabel("Äußere Iteration", fontsize=12)
    ax.set_ylabel(r"$\max(|V_{PV}| - V^{spec})$ [p.u.]", fontsize=12)
    # ax.set_title("Outer Convergence: Decoupled Q-Update\n" + r"$\Delta Q_k = \frac{|V^{spec}_k|^2 - |v_k|^2}{2 \cdot X_{kk}}$", fontsize=13)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=1e-7, top=1e-2)
    ax.set_xlim(1, 100)


    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def print_x_pp_matrices(save_name="x_pp_heatmaps.pgf"):
    """Compute and print X_pp matrices for sz_40_r010 and sz_40_r020 in LaTeX format, plus heatmap plot."""
    networks = get_salazar_scaling_networks()

    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.2), constrained_layout=True)

    im = None
    for idx, name in enumerate(["sz_40_r010", "sz_40_r020"]):
        print(f"\n=== {name} ===")
        info = networks[name]
        net_pandapower = info["constructor"]()
        network = build_network_from_pandapower(net_pandapower, include_pv=True)

        Z_B = np.linalg.inv(network.Y_dd)
        X = np.imag(Z_B)
        pv_idx = network.pv_indices
        X_pp = X[np.ix_(pv_idx, pv_idx)]

        n = X_pp.shape[0]
        print(r"\begin{bmatrix}")
        for i in range(n):
            row = " & ".join([f"{X_pp[i, j]:.4f}" for j in range(n)])
            end = r"\\" if i < n - 1 else ""
            print(f"  {row} {end}")
        print(r"\end{bmatrix}")

        # Plot heatmap – aspect='equal' => quadratische Zellen
        ax = axes[idx]
        im = ax.imshow(X_pp, cmap='viridis', aspect='equal')
        ax.set_title(rf"$n_{{\mathrm{{PV}}}}={network.n_pv}$", fontsize=11)
        ax.set_xlabel("PV Index")
        ax.set_ylabel("PV Index")

    # Gemeinsame Colorbar – constrained_layout platziert sie ohne Überlappung
    fig.colorbar(im, ax=axes, label=r'$X_{pp}$', shrink=0.8)

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)



def print_salazar_scaling_table():
    """Run solver on Salazar scaling suite (n <= 200) and print LaTeX table to console."""
    networks = get_salazar_scaling_networks()

    filtered = {k: v for k, v in networks.items() if v["n_bus_total"] <= 200}

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{|l|r|r|r|l|}")
    print(r"\hline")
    print(r"Netz & $n_{bus}$ & $n_{PV}$ & Iter. & Konvergiert \\")
    print(r"\hline")

    for name, info in filtered.items():
        # print(f"  Processing {name}...")
        name_escaped = name.replace("_", r"\_")

        try:
            net_pandapower = info["constructor"]()
            network = build_network_from_pandapower(net_pandapower, include_pv=True)
        except Exception as e:
            print(f"{name_escaped} & {info['n_bus_total']} & {info['n_pv']} & — & Fehler \\\\")
            print(f"    Warning: {e}")
            continue

        n_bus = network.n_bus_phases
        n_pv = network.n_pv if network.has_pv else 0

        if n_pv == 0:
            solver = TPFDenseSolver(tol=1e-8, max_iter=100)
        else:
            solver = TPFDensePVMethodA(
                tol=1e-8,
                max_iter_inner=100,
                max_iter_outer=100,
                tol_pv=1e-6,
                omega=1.0,
                use_decoupled=True
            )

        try:
            result = solver.solve(network)
        except Exception as e:
            print(f"{name_escaped} & {n_bus} & {n_pv} & — & Fehler \\\\")
            print(f"    Warning: {e}")
            continue

        if n_pv == 0:
            n_iter = result.iterations
        elif solver.pv_info is not None and solver.pv_info.pv_v_error_history:
            n_iter = len(solver.pv_info.pv_v_error_history)
        else:
            n_iter = 0

        conv = r"Ja" if result.converged else r"Nein"
        print(f"{name_escaped} & {n_bus} & {n_pv} & {n_iter} & {conv} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Ergebnisse: Salazar Skalierungsnetze (decoupled Q-update)}")
    print(r"\label{tab:salazar_scaling}")
    print(r"\end{table}")


def plot_max_pv_convergence(save_name="max_pv_convergence.pgf"):
    """Plot max PV nodes that converge vs network size."""
    from tpf.generators.network_generator_salazar import create_salazar_network

    network_sizes = [20, 40, 75, 120, 200, 350, 500, 750, 1000]
    max_pv_limit = 50
    seed = 42

    sizes_plot = []
    max_pv_converged = []

    for n_bus in network_sizes:
        print(f"  Testing n_bus={n_bus}...")
        max_conv = 0

        for n_pv in range(1, max_pv_limit + 1):
            try:
                net_pp = create_salazar_network(nodes=n_bus, n_pv=n_pv, seed=seed)
                network = build_network_from_pandapower(net_pp, include_pv=True)

                solver = TPFDensePVMethodA(
                    tol=1e-8, max_iter_inner=100, max_iter_outer=100,
                    tol_pv=1e-6, omega=1.0, use_decoupled=True
                )
                result = solver.solve(network)

                if result.converged:
                    max_conv = n_pv
                else:
                    break
            except Exception as e:
                print(f"    Warning at n_pv={n_pv}: {e}")
                break

        sizes_plot.append(n_bus)
        max_pv_converged.append(max_conv)
        print(f"    max converged: {max_conv}")

    fig, ax = plt.subplots(figsize=(5.91, 3))
    ax.loglog(sizes_plot, max_pv_converged, "o-", color="darkblue", linewidth=2, markersize=8)
    ax.set_xlabel("$n_{{\mathrm{{bus}}}}$", fontsize=12)
    ax.set_ylabel("Max $n_{{\mathrm{{PV}}}}$ konvergiert", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    # ax.set_xscale("log")
    ax.set_ylim(1, 1000)
    plt.tight_layout()
    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_outer_convergence_error_coupled(
    pv_counts: list = None,
    n_bus: int = 40,
    seed: int = 42,
    save_name: str = "outer_convergence_error_coupled.pgf"
):
    """
    Plot max(|V| - V_spec) vs outer iteration using COUPLED Q-update.
    Formula: ΔQ = 0.5 * X_pp^-1 * (|V_spec|² - |v|²)

    Parameters
    ----------
    pv_counts : list
        List of PV counts to test. Default: [0, 5, 10, 20, 30, 40]
    n_bus : int
        Network size (number of buses). Default: 40
    seed : int
        Random seed for reproducibility. Default: 42
    save_name : str
        Output filename
    """
    from tpf.generators.network_generator_salazar import create_salazar_network

    if pv_counts is None:
        pv_counts = [0, 5, 10, 20, 30, 40]

    fig, ax = plt.subplots(figsize=(5.91, 3.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pv_counts)))

    for idx, n_pv in enumerate(pv_counts):
        print(f"  Processing: n_bus={n_bus}, n_pv={n_pv}...")

        net_pp = create_salazar_network(nodes=n_bus, n_pv=n_pv, seed=seed)
        network = build_network_from_pandapower(net_pp, include_pv=True)

        if not network.has_pv or network.n_pv == 0:
            print(f"    Warning: No PV nodes, skipping")
            continue

        solver = TPFDensePVMethodA(
            tol=1e-8,
            max_iter_inner=100,
            max_iter_outer=100,
            tol_pv=1e-6,
            omega=1.0,
            use_decoupled=False  # COUPLED method
        )

        result = solver.solve(network)

        if solver.pv_info is None or not hasattr(solver.pv_info, 'pv_v_error_history'):
            print(f"    Warning: No PV error history")
            continue

        pv_errors = solver.pv_info.pv_v_error_history
        if not pv_errors:
            print(f"    Warning: Empty PV error history")
            continue

        linestyle = "x--" if not result.converged else "o--"
        iterations = np.arange(1, len(pv_errors) + 1)
        label = f"$n_{{PV}}={n_pv}$"
        ax.loglog(iterations, pv_errors, linestyle, color=colors[idx],
                  linewidth=1.5, markersize=6, label=label, alpha=0.8)

    ax.set_xlabel("Äußere Iteration", fontsize=12)
    ax.set_ylabel(r"$\max(|V_{PV}| - V^{spec})$ [p.u.]", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=1e-7, top=1e-2)
    ax.set_xlim(1, 10)

    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_inner_start_comparison(
    n_bus: int = 40,
    n_pv: int = 4,
    seed: int = 42,
    save_name: str = "inner_start_comparison.pgf"
):
    """
    Compare flat start (cold_start=True) vs warm start (cold_start=False)
    for the inner FPI loop in TPFDensePVMethodA.

    Parameters
    ----------
    n_bus : int
        Network size (number of buses). Default: 40
    n_pv : int
        Number of PV nodes. Default: 4
    seed : int
        Random seed for reproducibility. Default: 42
    save_name : str
        Output filename
    """
    from tpf.generators.network_generator_salazar import create_salazar_network

    print(f"  Creating network: n_bus={n_bus}, n_pv={n_pv}...")
    net_pp = create_salazar_network(nodes=n_bus, n_pv=n_pv, seed=seed)
    network = build_network_from_pandapower(net_pp, include_pv=True)

    print(f"  Running with cold_start=True (flat start)...")
    solver_cold = TPFDensePVMethodA(
        tol=1e-8,
        max_iter_inner=50,
        max_iter_outer=100,
        tol_pv=1e-8,
        omega=1.0,
        use_decoupled=True,
        cold_start=True
    )
    print(f"    Solver created, calling solve()...")
    result_cold = solver_cold.solve(network)
    print(f"    solve() completed.")

    print(f"  Running with cold_start=False (warm start)...")
    solver_warm = TPFDensePVMethodA(
        tol=1e-8,
        max_iter_inner=50,
        max_iter_outer=100,
        tol_pv=1e-8,
        omega=1.0,
        use_decoupled=True,
        cold_start=False
    )
    print(f"    Solver created, calling solve()...")
    result_warm = solver_warm.solve(network)
    print(f"    solve() completed.")

    pv_errors_cold = solver_cold.pv_info.pv_v_error_history if solver_cold.pv_info else []
    pv_errors_warm = solver_warm.pv_info.pv_v_error_history if solver_warm.pv_info else []

    inner_v_change_cold = solver_cold.pv_info.inner_v_change_all if solver_cold.pv_info else []
    inner_v_change_warm = solver_warm.pv_info.inner_v_change_all if solver_warm.pv_info else []

    outer_starts_cold = solver_cold.pv_info.outer_start_indices if solver_cold.pv_info else []
    outer_starts_warm = solver_warm.pv_info.outer_start_indices if solver_warm.pv_info else []

    time_cold_ms = solver_cold.pv_info.total_inner_fpi_time_ms if solver_cold.pv_info else 0
    time_warm_ms = solver_warm.pv_info.total_inner_fpi_time_ms if solver_warm.pv_info else 0

    inner_iters_cold = solver_cold.pv_info.inner_iterations_per_outer if solver_cold.pv_info else []
    inner_iters_warm = solver_warm.pv_info.inner_iterations_per_outer if solver_warm.pv_info else []

    total_inner_cold = sum(inner_iters_cold) if inner_iters_cold else 0
    total_inner_warm = sum(inner_iters_warm) if inner_iters_warm else 0

    speedup = time_cold_ms / time_warm_ms if time_warm_ms > 0 else 0

    def compute_peak_decline(inner_v_change_all, outer_start_indices):
        if not inner_v_change_all or not outer_start_indices or len(outer_start_indices) < 2:
            return np.nan
        peaks = []
        for i, start_idx in enumerate(outer_start_indices):
            if i + 1 < len(outer_start_indices):
                end_idx = outer_start_indices[i + 1]
            else:
                end_idx = len(inner_v_change_all)
            if start_idx < len(inner_v_change_all) and end_idx <= len(inner_v_change_all):
                peak = inner_v_change_all[end_idx - 1]
                peaks.append(peak)
        if len(peaks) < 2:
            return np.nan
        declines = []
        for i in range(len(peaks) - 1):
            if peaks[i] > 1e-15:
                declines.append(peaks[i + 1] / peaks[i])
        return np.mean(declines) if declines else np.nan

    peak_decline_cold = compute_peak_decline(inner_v_change_cold, outer_starts_cold)
    peak_decline_warm = compute_peak_decline(inner_v_change_warm, outer_starts_warm)

    rho_diag_cold = compute_spectral_radius_diagonal(network, omega=1.0) if network.has_pv and network.n_pv > 0 else np.nan
    rho_diag_warm = rho_diag_cold

    rho_corr_cold = compute_spectral_radius_corrected(network, omega=1.0) if network.has_pv and network.n_pv > 0 else np.nan
    rho_corr_warm = rho_corr_cold

    kappa_cold = compute_empirical_contraction(pv_errors_cold) if pv_errors_cold else np.nan
    kappa_warm = compute_empirical_contraction(pv_errors_warm) if pv_errors_warm else np.nan

    print(f"\n  Results:")
    print(f"    Cold start: {time_cold_ms:.2f} ms, {total_inner_cold} inner iterations")
    print(f"    Warm start: {time_warm_ms:.2f} ms, {total_inner_warm} inner iterations")
    print(f"    Speedup: {speedup:.2f}x")

    print(f"\n  Convergence Metrics:")
    print(f"  ║  Convergence Metrics: Cold vs Warm Start                     ║'")
    print(f"  ║  Start                 ║ ρ_diag   ║ ρ_corr   ║ κ        ║Decl║'")
    print(f"  ║  Cold                  ║ {rho_diag_cold:8.4f} ║ {rho_corr_cold:8.4f} ║ {kappa_cold:8.4f} ║{peak_decline_cold:5.2f}║'")
    print(f"  ║  Warm                  ║ {rho_diag_warm:8.4f} ║ {rho_corr_warm:8.4f} ║ {kappa_warm:8.4f} ║{peak_decline_warm:5.2f}║'")

    fig, axs = plt.subplots(2, 1, figsize=(5.91, 3.5))
    ax1 = axs[0]
    ax3 = axs[1]

    x_cold = list(range(1, len(inner_v_change_cold) + 1))
    x_warm = list(range(1, len(inner_v_change_warm) + 1))

    ax1.semilogy(x_cold, inner_v_change_cold, "s-", color="darkblue",
                 linewidth=1.0, markersize=2, alpha=0.8)

    # ax1.set_xlabel("Kumulative Inner-Iteration", fontsize=12)
    ax1.set_ylabel(r"Kalt", fontsize=12)
    # ax1.set_title("Kaltstart (flat)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(bottom=1e-12, top=1)

    ax3.semilogy(x_warm, inner_v_change_warm, "s-", color="darkred",
                 linewidth=1.0, markersize=2, alpha=0.8)

    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_ylabel(r"", fontsize=12)
    # ax3.set_title("Warm", fontsize=12, fontweight="bold")
    ax3.set_ylabel(r"Warm", fontsize=12)

    ax3.grid(True, which="both", alpha=0.3)
    ax3.set_ylim(bottom=1e-12, top=1)
    right = 110
    ax1.set_xlim(left=0, right=right)
    ax3.set_xlim(left=0, right=right)

    fig.supylabel(r"$\max(|V_{neu}| - |V_{alt}|)$ [p.u.]")
    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_adaptive_inner_comparison(
    n_bus: int = 40,
    n_pv: int = 4,
    seed: int = 42,
    save_name: str = "adaptive_inner_comparison.pgf"
):
    """
    Compare fixed inner iteration control (adaptive_inner=False) vs
    adaptive inner iteration control (adaptive_inner=True) in TPFDensePVMethodA.

    Parameters
    ----------
    n_bus : int
        Network size (number of buses). Default: 40
    n_pv : int
        Number of PV nodes. Default: 4
    seed : int
        Random seed for reproducibility. Default: 42
    save_name : str
        Output filename
    """
    from tpf.generators.network_generator_salazar import create_salazar_network

    print(f"  Creating network: n_bus={n_bus}, n_pv={n_pv}...")
    net_pp = create_salazar_network(nodes=n_bus, n_pv=n_pv, seed=seed)
    network = build_network_from_pandapower(net_pp, include_pv=True)

    # networks = get_salazar_scaling_networks()
    # info = networks['sz_40_r010']
    # net_pandapower = info["constructor"]()
    # network = build_network_from_pandapower(net_pandapower, include_pv=True)

    print(f"  Running with adaptive_inner=False (fixed inner iterations)...")
    solver_fixed = TPFDensePVMethodA(
        tol=1e-8,
        max_iter_inner=20,
        max_iter_outer=50,
        tol_pv=1e-6,
        omega=1.0,
        use_decoupled=True,
        cold_start=False,
        adaptive_inner=False
    )
    print(f"    Solver created, calling solve()...")
    result_fixed = solver_fixed.solve(network)
    print(f"    solve() completed.")

    print(f"  Running with adaptive_inner=True (adaptive tolerance)...")
    solver_adaptive = TPFDensePVMethodA(
        tol=1e-8,
        max_iter_inner=20,
        max_iter_outer=50,
        tol_pv=1e-6,
        omega=1.0,
        use_decoupled=True,
        cold_start=False,
        adaptive_inner=True
    )
    print(f"    Solver created, calling solve()...")
    result_adaptive = solver_adaptive.solve(network)
    print(f"    solve() completed.")

    pv_errors_fixed = solver_fixed.pv_info.pv_v_error_history if solver_fixed.pv_info else []
    pv_errors_adaptive = solver_adaptive.pv_info.pv_v_error_history if solver_adaptive.pv_info else []

    inner_v_change_fixed = solver_fixed.pv_info.inner_v_change_all if solver_fixed.pv_info else []
    inner_v_change_adaptive = solver_adaptive.pv_info.inner_v_change_all if solver_adaptive.pv_info else []

    outer_starts_fixed = solver_fixed.pv_info.outer_start_indices if solver_fixed.pv_info else []
    outer_starts_adaptive = solver_adaptive.pv_info.outer_start_indices if solver_adaptive.pv_info else []

    time_fixed_ms = result_fixed.elapsed_time_s * 1000
    time_adaptive_ms = result_adaptive.elapsed_time_s * 1000

    inner_iters_fixed = solver_fixed.pv_info.inner_iterations_per_outer if solver_fixed.pv_info else []
    inner_iters_adaptive = solver_adaptive.pv_info.inner_iterations_per_outer if solver_adaptive.pv_info else []

    total_inner_fixed = sum(inner_iters_fixed) if inner_iters_fixed else 0
    total_inner_adaptive = sum(inner_iters_adaptive) if inner_iters_adaptive else 0

    speedup = time_fixed_ms / time_adaptive_ms if time_adaptive_ms > 0 else 0

    def compute_peak_decline(inner_v_change_all, outer_start_indices):
        if not inner_v_change_all or not outer_start_indices or len(outer_start_indices) < 2:
            return np.nan
        peaks = []
        for i, start_idx in enumerate(outer_start_indices):
            if i + 1 < len(outer_start_indices):
                end_idx = outer_start_indices[i + 1]
            else:
                end_idx = len(inner_v_change_all)
            if start_idx < len(inner_v_change_all) and end_idx <= len(inner_v_change_all):
                peak = inner_v_change_all[end_idx - 1]
                peaks.append(peak)
        if len(peaks) < 2:
            return np.nan
        declines = []
        for i in range(len(peaks) - 1):
            if peaks[i] > 1e-15:
                declines.append(peaks[i + 1] / peaks[i])
        return np.mean(declines) if declines else np.nan

    peak_decline_fixed = compute_peak_decline(inner_v_change_fixed, outer_starts_fixed)
    peak_decline_adaptive = compute_peak_decline(inner_v_change_adaptive, outer_starts_adaptive)

    rho_diag = compute_spectral_radius_diagonal(network, omega=1.0) if network.has_pv and network.n_pv > 0 else np.nan
    rho_corr = compute_spectral_radius_corrected(network, omega=1.0) if network.has_pv and network.n_pv > 0 else np.nan

    kappa_fixed = compute_empirical_contraction(pv_errors_fixed) if pv_errors_fixed else np.nan
    kappa_adaptive = compute_empirical_contraction(pv_errors_adaptive) if pv_errors_adaptive else np.nan

    print(f"\n  Results:")
    print(f"    Fixed (adaptive_inner=False): {time_fixed_ms:.2f} ms, {total_inner_fixed} inner iterations, {len(pv_errors_fixed)} outer iterations")
    print(f"    Adaptive (adaptive_inner=True): {time_adaptive_ms:.2f} ms, {total_inner_adaptive} inner iterations, {len(pv_errors_adaptive)} outer iterations")
    print(f"    Speedup (fixed/adaptive): {speedup:.2f}x")

    print(f"\n  Convergence Metrics:")
    print(f"  ║  Method                 ║ ρ_diag   ║ ρ_corr   ║ κ        ║Decl║")
    print(f"  ║  Fixed                  ║ {rho_diag:8.4f} ║ {rho_corr:8.4f} ║ {kappa_fixed:8.4f} ║{peak_decline_fixed:5.2f}║")
    print(f"  ║  Adaptive               ║ {rho_diag:8.4f} ║ {rho_corr:8.4f} ║ {kappa_adaptive:8.4f} ║{peak_decline_adaptive:5.2f}║")

    fig, axs = plt.subplots(2, 1, figsize=(5.91, 3.5))
    ax1 = axs[0]
    ax2 = axs[1]

    x_fixed = list(range(1, len(inner_v_change_fixed) + 1))
    x_adaptive = list(range(1, len(inner_v_change_adaptive) + 1))

    right = max(len(x_fixed), len(x_adaptive), 55)

    ax1.semilogy(x_fixed, inner_v_change_fixed, "s-", color="darkblue",
                 linewidth=1.0, markersize=2, alpha=0.8)

    ax1.set_ylabel(r"Fixiert", fontsize=12)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(bottom=1e-11, top=1)
    ax1.set_xlim(left=0, right=right)

    ax2.semilogy(x_adaptive, inner_v_change_adaptive, "s-", color="darkred",
                 linewidth=1.0, markersize=2, alpha=0.8)

    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel(r"Adaptiv", fontsize=12)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(bottom=1e-11, top=1)
    ax2.set_xlim(left=0, right=right+5)

    fig.supylabel(r"$\max(|V_{neu}| - |V_{alt}|)$ [p.u.]")
    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_salazar_scaling_comparison(
    max_n_bus: int = 40,
    save_name: str = "salazar_scaling_speedup.pgf"
):
    """
    Run Salazar scaling suite with cold start vs warm start using run_validation_suite.
    Generate boxplot of speedup per network size and LaTeX table to console.

    Parameters
    ----------
    max_n_bus : int
        Maximum network size to include. Default: 1000
    save_name : str
        Output filename for boxplot
    """
    networks = get_salazar_scaling_networks()
    filtered = {k: v for k, v in networks.items() if v["n_bus_total"] <= max_n_bus}

    sizes = sorted(set(v["n_bus_total"] for v in filtered.values()))
    print(f"\n  Network sizes to process: {sizes}")

    print(f"  Running cold start (flat start) validation suite for {len(filtered)} networks...")
    records_cold = run_validation_suite(
        filtered, omega=1.0, tol_pass=1e-6, verbose=False,
        cold_start=True, analysis="full", adaptive_inner=False, sparse=False
    )
    print(f"    Cold start: {len(records_cold)} records")

    print(f"  Running warm start validation suite for {len(filtered)} networks...")
    records_warm = run_validation_suite(
        filtered, omega=1.0, tol_pass=1e-6, verbose=False,
        cold_start=False, analysis="full", adaptive_inner=False, sparse=False
    )
    print(f"    Warm start: {len(records_warm)} records")

    records_cold_by_name = {r["name"]: r for r in records_cold}
    records_warm_by_name = {r["name"]: r for r in records_warm}

    results_by_size = {s: [] for s in sizes}
    all_results = []

    for name in sorted(filtered.keys()):
        info = filtered[name]
        n_bus_total = info["n_bus_total"]

        rec_cold = records_cold_by_name.get(name)
        rec_warm = records_warm_by_name.get(name)

        if rec_cold is None or rec_warm is None:
            print(f"    Warning: Missing record for {name}")
            continue

        n_pv = rec_cold.get("n_pv", 0)
        time_cold_ms = rec_cold.get("tpf_time_ms", 0)
        time_warm_ms = rec_warm.get("tpf_time_ms", 0)
        inner_cold = rec_cold.get("tpf_inner_iter_total", 0)
        inner_warm = rec_warm.get("tpf_inner_iter_total", 0)
        outer_cold = rec_cold.get("tpf_outer_iter", 0)
        outer_warm = rec_warm.get("tpf_outer_iter", 0)
        converged = rec_cold.get("tpf_converged", False) and rec_warm.get("tpf_converged", False)

        speedup = time_cold_ms / time_warm_ms if time_warm_ms > 0 else 0

        all_results.append({
            "name": name,
            "n_bus": n_bus_total,
            "n_pv": n_pv,
            "inner_cold": inner_cold,
            "inner_warm": inner_warm,
            "outer_cold": outer_cold,
            "outer_warm": outer_warm,
            "time_cold_ms": time_cold_ms,
            "time_warm_ms": time_warm_ms,
            "speedup": speedup,
            "converged": converged,
        })

        results_by_size[n_bus_total].append({
            "name": name,
            "n_pv": n_pv,
            "speedup": speedup,
            "inner_cold": inner_cold,
            "inner_warm": inner_warm,
            "outer_cold": outer_cold,
            "outer_warm": outer_warm,
            "time_cold_ms": time_cold_ms,
            "time_warm_ms": time_warm_ms,
            "converged": converged,
        })

    print(f"\n  Generating LaTeX table...")

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{r|r|r|r|r|r|r|r}")
    print(r"\hline")
    print(r"$n_{bus}$ & $n_{PV}$ & Inner & Outer & $t_{cold}$ (ms) & $t_{warm}$ (ms) & Speedup & Conv\\")
    print(r"\hline")

    for r in all_results:
        conv_str = r"Ja" if r["converged"] else r"Nein"
        speedup_str = f"{r['speedup']:.2f}" if r["speedup"] > 0 else "—"
        print(f"{r['n_bus']} & {r['n_pv']} & {r['inner_cold']} & {r['outer_cold']} & {r['time_cold_ms']:.1f} & {r['time_warm_ms']:.1f} & {speedup_str} & {conv_str}\\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Salazar Skalierungsnetze: Kaltstart vs Warmstart}")
    print(r"\label{tab:salazar_scaling_cold_warm}")
    print(r"\end{table}")

    print(f"\n  Generating boxplot...")

    speedup_data = []
    labels = []
    for size in sizes:
        if results_by_size[size]:
            speedups = [r["speedup"] for r in results_by_size[size] if r["speedup"] > 0]
            if speedups:
                speedup_data.append(speedups)
                labels.append(str(size))

    fig, ax = plt.subplots(figsize=(5.91, 3.5))

    bp = ax.boxplot(speedup_data, tick_labels=labels, patch_artist=True)

    for box in bp["boxes"]:
        box.set_facecolor("lightblue")
        box.set_edgecolor("darkblue")

    for median in bp["medians"]:
        median.set_color("darkred")
        median.set_linewidth(2)

    means = [np.mean(d) for d in speedup_data]
    ax.scatter(range(1, len(means) + 1), means, color="green", marker="D", s=50, zorder=5, label="Mittelwert")

    ax.set_xlabel("$n_{bus}$", fontsize=12)
    ax.set_ylabel("Speedup ($t_{cold} / t_{warm}$)", fontsize=12)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_salazar_adaptive_speedup(
    max_n_bus: int = 1000,
    save_name: str = "salazar_adaptive_speedup.pgf"
):
    """
    Run Salazar scaling suite with adaptive_inner=False vs adaptive_inner=True.
    Generate boxplot of speedup per network size and LaTeX table to console.

    Parameters
    ----------
    max_n_bus : int
        Maximum network size to include. Default: 40
    save_name : str
        Output filename for boxplot
    """
    networks = get_salazar_scaling_networks()
    filtered = {k: v for k, v in networks.items() if v["n_bus_total"] <= max_n_bus and v["n_pv"]>0}

    sizes = sorted(set(v["n_bus_total"] for v in filtered.values()))
    print(f"\n  Network sizes to process: {sizes}")

    print(f"  Running with adaptive_inner=False (fixed tolerance)...")
    records_fixed = run_validation_suite(
        filtered, omega=1.0, tol_pass=1e-6, verbose=False,
        cold_start=False, analysis="full", adaptive_inner=False, sparse=False
    )
    print(f"    Fixed inner: {len(records_fixed)} records")

    print(f"  Running with adaptive_inner=True (adaptive tolerance)...")
    records_adaptive = run_validation_suite(
        filtered, omega=1.0, tol_pass=1e-6, verbose=False,
        cold_start=False, analysis="full", adaptive_inner=True, sparse=False
    )
    print(f"    Adaptive inner: {len(records_adaptive)} records")

    records_fixed_by_name = {r["name"]: r for r in records_fixed}
    records_adaptive_by_name = {r["name"]: r for r in records_adaptive}

    results_by_size = {s: [] for s in sizes}
    all_results = []

    for name in sorted(filtered.keys()):
        info = filtered[name]
        n_bus_total = info["n_bus_total"]

        rec_fixed = records_fixed_by_name.get(name)
        rec_adaptive = records_adaptive_by_name.get(name)

        if rec_fixed is None or rec_adaptive is None:
            print(f"    Warning: Missing record for {name}")
            continue

        n_pv = rec_fixed.get("n_pv", 0)
        time_fixed_ms = rec_fixed.get("tpf_time_ms", 0)
        time_adaptive_ms = rec_adaptive.get("tpf_time_ms", 0)
        inner_fixed = rec_fixed.get("tpf_inner_iter_total", 0)
        inner_adaptive = rec_adaptive.get("tpf_inner_iter_total", 0)
        outer_fixed = rec_fixed.get("tpf_outer_iter", 0)
        outer_adaptive = rec_adaptive.get("tpf_outer_iter", 0)
        converged = rec_fixed.get("tpf_converged", False) and rec_adaptive.get("tpf_converged", False)

        total_iter_fixed = inner_fixed + outer_fixed
        total_iter_adaptive = inner_adaptive + outer_adaptive
        speedup = total_iter_fixed / total_iter_adaptive if total_iter_adaptive > 0 else 0

        all_results.append({
            "name": name,
            "n_bus": n_bus_total,
            "n_pv": n_pv,
            "inner_fixed": inner_fixed,
            "inner_adaptive": inner_adaptive,
            "outer_fixed": outer_fixed,
            "outer_adaptive": outer_adaptive,
            "time_fixed_ms": time_fixed_ms,
            "time_adaptive_ms": time_adaptive_ms,
            "speedup": speedup,
            "converged": converged,
        })

        results_by_size[n_bus_total].append({
            "name": name,
            "n_pv": n_pv,
            "speedup": speedup,
            "inner_fixed": inner_fixed,
            "inner_adaptive": inner_adaptive,
            "outer_fixed": outer_fixed,
            "outer_adaptive": outer_adaptive,
            "time_fixed_ms": time_fixed_ms,
            "time_adaptive_ms": time_adaptive_ms,
            "converged": converged,
        })

    print(f"\n  Generating LaTeX table...")

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{r|r|r|r|r|r|r|r}")
    print(r"\hline")
    print(r"$n_{bus}$ & $n_{PV}$ & Inner$_{fix}$ & Outer$_{fix}$ & Inner$_{adj}$ & Outer$_{adj}$ & Speedup & Conv\\")
    print(r"\hline")

    for r in all_results:
        conv_str = r"Ja" if r["converged"] else r"Nein"
        speedup_str = f"{r['speedup']:.2f}" if r["speedup"] > 0 else "—"
        print(f"{r['n_bus']} & {r['n_pv']} & {r['inner_fixed']} & {r['outer_fixed']} & {r['inner_adaptive']} & {r['outer_adaptive']} & {speedup_str} & {conv_str}\\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Salazar Skalierungsnetze: Fixe vs Adaptive innere Iteration}")
    print(r"\label{tab:salazar_adaptive_speedup}")
    print(r"\end{table}")

    print(f"\n  Generating boxplot...")

    speedup_data = []
    labels = []
    for size in sizes:
        if results_by_size[size]:
            speedups = [r["speedup"] for r in results_by_size[size] if r["speedup"] > 0]
            if speedups:
                speedup_data.append(speedups)
                labels.append(str(size))

    fig, ax = plt.subplots(figsize=(5.91, 3.5))

    bp = ax.boxplot(speedup_data, tick_labels=labels, patch_artist=True)

    for box in bp["boxes"]:
        box.set_facecolor("white")
        box.set_edgecolor("black")

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    means = [np.mean(d) for d in speedup_data]
    ax.scatter(range(1, len(means) + 1), means, color="black", marker="D", s=20, zorder=5, label="Mittelwert")

    ax.set_xlabel("$n_{\mathrm{bus}}$", fontsize=12)
    ax.set_ylabel("$Iter_{\mathrm{fix}} / Iter_{\mathrm{adapt}}$", fontsize=12)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=1)
    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_pq_scaling_time(
    max_n_bus: int = 1000,
    save_name: str = "pq_scaling_time.pgf"
):
    """
    Run Salazar scaling suite with n_pv=0 (PQ-only networks) using TPFDenseSolver.
    Generate log-log plot of computing time vs network size and LaTeX table to console.

    Parameters
    ----------
    max_n_bus : int
        Maximum network size to include. Default: 1000
    save_name : str
        Output filename for plot
    """
    networks = get_salazar_scaling_networks()
    filtered = {k: v for k, v in networks.items() if v["n_bus_total"] <= max_n_bus and v["n_pv"] == 0}

    sizes = sorted(set(v["n_bus_total"] for v in filtered.values()))
    print(f"\n  Network sizes to process (n_pv=0): {sizes}")

    results_by_size = {s: [] for s in sizes}
    all_results = []

    for name, info in filtered.items():
        n_bus_total = info["n_bus_total"]
        print(f"  Processing {name}...")

        try:
            net_pandapower = info["constructor"]()
            network = build_network_from_pandapower(net_pandapower, include_pv=True)
        except Exception as e:
            print(f"    Warning: Failed to build network: {e}")
            continue

        solver = TPFDenseSolver(tol=1e-8, max_iter=100)

        try:
            result = solver.solve(network)
        except Exception as e:
            print(f"    Warning: Solver failed: {e}")
            continue

        time_ms = result.elapsed_time_s * 1000
        iterations = result.iterations
        converged = result.converged

        all_results.append({
            "name": name,
            "n_bus": n_bus_total,
            "time_ms": time_ms,
            "iterations": iterations,
            "converged": converged,
        })

        results_by_size[n_bus_total].append({
            "name": name,
            "time_ms": time_ms,
            "iterations": iterations,
            "converged": converged,
        })

    print(f"\n  Generating LaTeX table...")

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{r|r|r|r|r}")
    print(r"\hline")
    print(r"$n_{bus}$ & Zeit (ms) & Iter. & Konv. \\")
    print(r"\hline")

    for r in all_results:
        conv_str = r"Ja" if r["converged"] else r"Nein"
        print(f"{r['n_bus']} & {r['time_ms']:.2f} & {r['iterations']} & {conv_str}\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Salazar Skalierungsnetze: PQ-Netze (n$_{PV}$=0)}")
    print(r"\label{tab:pq_scaling_time}")
    print(r"\end{table}")

    print(f"\n  Generating plot...")

    mean_times = []
    labels = []
    for size in sizes:
        if results_by_size[size]:
            times = [r["time_ms"] for r in results_by_size[size] if r["time_ms"] > 0]
            if times:
                mean_times.append(np.mean(times))
                labels.append(str(size))

    fig, ax = plt.subplots(figsize=(5.91, 3.5))

    x_vals = [int(l) for l in labels]
    ax.loglog(x_vals, mean_times, "o-", color="black", linewidth=1.5, markersize=6)

    ax.set_xlabel("$n_{\mathrm{bus}}$", fontsize=12)
    ax.set_ylabel("Zeit (ms)", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_timing_vs_size(save_name="timing_vs_size.pgf"):
    """
    Plot (c): Solver time vs network size (n_bus) - boxplot.

    Uses validation suite on Salazar scaling networks to compare
    NR, TPF (dense), and TPF (sparse) solver times.
    """
    from collections import defaultdict

    print("\n  Running validation suite for timing vs size plot...")
    networks = get_salazar_scaling_networks()
    omega = 1.0

    records = run_validation_suite(networks, omega=omega, verbose=True, sparse=True)

    timing_data = [r for r in records
                   if r.get("nr_converged")
                   and r.get("tpf_converged")
                   and r.get("nr_time_ms", 0) > 0
                   and r.get("tpf_time_ms", 0) > 0]

    timing_data_sparse = [r for r in records
                          if r.get("nr_converged")
                          and r.get("sparse_converged")
                          and r.get("nr_time_ms", 0) > 0
                          and r.get("sparse_time_ms", 0) > 0]

    if not timing_data:
        print("  ! No timing data available.")
        return

    nr_by_size = defaultdict(list)
    tpf_by_size = defaultdict(list)
    for r in timing_data:
        nr_by_size[r["n_bus"]].append(r["nr_time_ms"])
        tpf_by_size[r["n_bus"]].append(r["tpf_time_ms"])

    sparse_by_size = defaultdict(list)
    for r in timing_data_sparse:
        sparse_by_size[r["n_bus"]].append(r["sparse_time_ms"])

    sizes = sorted(nr_by_size.keys())

    fig, ax = plt.subplots(figsize=(5.91, 3.5))

    box_width = 0.18
    positions_nr = np.arange(len(sizes)) - box_width
    positions_tpf = np.arange(len(sizes))
    positions_sparse = np.arange(len(sizes)) + box_width

    ax.boxplot(
        [nr_by_size[s] for s in sizes],
        positions=positions_nr,
        widths=box_width * 0.8,
        patch_artist=True,
        boxprops=dict(facecolor="tab:red", alpha=0.6),
        medianprops=dict(color="darkred", linewidth=1.5),
        whiskerprops=dict(color="tab:red", linewidth=1.2),
        capprops=dict(color="tab:red", linewidth=1.2),
        flierprops=dict(marker="s", markerfacecolor="tab:red",
                        markersize=4, alpha=0.6),
    )

    ax.boxplot(
        [tpf_by_size[s] for s in sizes],
        positions=positions_tpf,
        widths=box_width * 0.8,
        patch_artist=True,
        boxprops=dict(facecolor="tab:blue", alpha=0.6),
        medianprops=dict(color="darkblue", linewidth=1.5),
        whiskerprops=dict(color="tab:blue", linewidth=1.2),
        capprops=dict(color="tab:blue", linewidth=1.2),
        flierprops=dict(marker="o", markerfacecolor="tab:blue",
                        markersize=4, alpha=0.6),
    )

    # ax.boxplot(
    #     [sparse_by_size[s] for s in sizes],
    #     positions=positions_sparse,
    #     widths=box_width * 0.8,
    #     patch_artist=True,
    #     boxprops=dict(facecolor="tab:green", alpha=0.6),
    #     medianprops=dict(color="darkgreen", linewidth=1.5),
    #     whiskerprops=dict(color="tab:green", linewidth=1.2),
    #     capprops=dict(color="tab:green", linewidth=1.2),
    #     flierprops=dict(marker="^", markerfacecolor="tab:green",
    #                     markersize=4, alpha=0.6),
    # )

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_yscale("log")

    ax.set_xlabel(r"$n_{\mathrm{bus}}$", fontsize=12)
    ax.set_ylabel("Rechenzeit [ms]", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

    ax.legend([
        plt.Rectangle((0, 0), 1, 1, fc="tab:red", alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, fc="tab:green", alpha=0.6),
    ], ["NR (pandapower)", "TPF Methode A (dense)", "TPF Methode A (sparse)"],
       fontsize=9, loc="upper left")

    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


def plot_timing_vs_pv_ratio(save_name="timing_vs_pv_ratio.pgf"):
    """
    Plot (d): Solver time vs PV ratio (n_pv/n_bus) - scatter plot.

    Uses validation suite on Salazar scaling networks to compare
    NR, TPF (dense), and TPF (sparse) solver times vs PV penetration.
    """
    from collections import defaultdict

    print("\n  Running validation suite for timing vs PV ratio plot...")
    networks = get_salazar_scaling_networks()
    omega = 1.0

    records = run_validation_suite(networks, omega=omega, verbose=True, sparse=True)

    timing_data = [r for r in records
                   if r.get("nr_converged")
                   and r.get("tpf_converged")
                   and r.get("nr_time_ms", 0) > 0
                   and r.get("tpf_time_ms", 0) > 0]

    timing_data_sparse = [r for r in records
                          if r.get("nr_converged")
                          and r.get("sparse_converged")
                          and r.get("nr_time_ms", 0) > 0
                          and r.get("sparse_time_ms", 0) > 0]

    if not timing_data:
        print("  ! No timing data available.")
        return

    all_timing = timing_data + timing_data_sparse
    n_bus_arr = np.array([r["n_bus"] for r in all_timing])

    size_norm = (n_bus_arr - n_bus_arr.min()) / max(n_bus_arr.max() - n_bus_arr.min(), 1)
    size_cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(5.91, 3.5))

    pv_ratios = np.array([r.get("pv_ratio", r["n_pv"] / max(r["n_bus"], 1))
                          for r in all_timing])
    nr_times_r = np.array([r["nr_time_ms"] for r in all_timing])

    ax.scatter(
        pv_ratios * 100, nr_times_r,
        c=size_norm, cmap=size_cmap, marker="s", s=60,
        edgecolors="tab:red", linewidths=1.5, zorder=5, alpha=0.8,
        label="NR (pandapower)",
    )

    if timing_data:
        tpf_times_r = np.array([r["tpf_time_ms"] for r in timing_data])
        tpf_ratios = np.array([r.get("pv_ratio", r["n_pv"] / max(r["n_bus"], 1))
                               for r in timing_data])
        tpf_n_bus = np.array([r["n_bus"] for r in timing_data])
        tpf_size_norm = (tpf_n_bus - tpf_n_bus.min()) / max(tpf_n_bus.max() - tpf_n_bus.min(), 1)

        ax.scatter(
            tpf_ratios * 100, tpf_times_r,
            c=tpf_size_norm, cmap=size_cmap, marker="o", s=60,
            edgecolors="tab:blue", linewidths=1.5, zorder=5, alpha=0.8,
            label="TPF Methode A",
        )

    if timing_data_sparse:
        sparse_times_r = np.array([r["sparse_time_ms"] for r in timing_data_sparse])
        sparse_ratios = np.array([r.get("pv_ratio", r["n_pv"] / max(r["n_bus"], 1))
                                  for r in timing_data_sparse])
        sparse_n_bus = np.array([r["n_bus"] for r in timing_data_sparse])
        sparse_size_norm = (sparse_n_bus - sparse_n_bus.min()) / max(sparse_n_bus.max() - sparse_n_bus.min(), 1)

        ax.scatter(
            sparse_ratios * 100, sparse_times_r,
            c=sparse_size_norm, cmap=size_cmap, marker="^", s=70,
            edgecolors="tab:green", linewidths=1.5, zorder=5, alpha=0.9,
            label="TPF Sparse",
        )

    nr_by_size = defaultdict(list)
    tpf_by_size = defaultdict(list)
    for r in timing_data:
        ratio = r.get("pv_ratio", r["n_pv"] / max(r["n_bus"], 1))
        nr_by_size[r["n_bus"]].append((ratio, r["nr_time_ms"]))
        tpf_by_size[r["n_bus"]].append((ratio, r["tpf_time_ms"]))

    for size in sorted(nr_by_size.keys()):
        nr_points = sorted(nr_by_size[size], key=lambda x: x[0])
        nr_x = [p[0] * 100 for p in nr_points]
        nr_y = [p[1] for p in nr_points]
        if len(nr_x) > 1:
            ax.plot(nr_x, nr_y, color="tab:red", linestyle="-", linewidth=1.0, alpha=0.5, zorder=2)

        tpf_points = sorted(tpf_by_size[size], key=lambda x: x[0])
        tpf_x = [p[0] * 100 for p in tpf_points]
        tpf_y = [p[1] for p in tpf_points]
        if len(tpf_x) > 1:
            ax.plot(tpf_x, tpf_y, color="tab:blue", linestyle="-", linewidth=1.0, alpha=0.5, zorder=2)

        if tpf_y:
            max_idx = np.argmax(tpf_y)
            ax.annotate(
                f"n={size}",
                (tpf_x[max_idx], tpf_y[max_idx]),
                fontsize=7, color="tab:blue", fontweight="bold",
                textcoords="offset points", xytext=(5, 0),
            )

    ax.set_yscale("log")
    ax.set_xlabel("PV-Durchdringung: $n_{PV} / n_{total}$ [%]", fontsize=12)
    ax.set_ylabel("Rechenzeit [ms]", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    cbar = plt.colorbar(ax, pad=0.02, fraction=0.04)
    cbar.set_label(r"$n_{\mathrm{bus}}$", fontsize=10)
    tick_vals = np.linspace(0, 1, 5)
    tick_labels = [f"{int(n_bus_arr.min() + t * (n_bus_arr.max() - n_bus_arr.min()))}"
                   for t in tick_vals]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(tick_labels)

    plt.tight_layout()

    save_path = SAVE_DIR / save_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")

    if not USE_PGF:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # print_salazar_scaling_table()
    # plot_max_pv_convergence()
    # print_x_pp_matrices()
    # plot_outer_convergence_error_coupled()
    # plot_inner_start_comparison()
    # plot_adaptive_inner_comparison()
    # plot_salazar_scaling_comparison()
    # plot_salazar_adaptive_speedup()
    # plot_pq_scaling_time()
    # plot_baseline_tpf_vs_nr()
    plot_timing_vs_size()
    # plot_timing_vs_pv_ratio()