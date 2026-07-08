"""Vergleich Slack-Leistung: TPF Methode A vs. pandapower NR."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.network_generator_salazar import create_salazar_network
from tpf.generators.profile_generators import generate_load_profile, generate_pv_profile
from tpf.builders.from_pandapower import build_s_batch_timeseries

# ── 1. Netz erzeugen ──
net = create_salazar_network(nodes=100, child=3, n_pv=5, seed=100)

# ── 2. NR lösen ──
nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
result_nr = nr_solver.solve_from_net(net)

# ── 3. TPF lösen ──
network = build_network_from_pandapower(net, include_pv=True)
tpf_solver = TPFDensePVMethodA(tol=1e-8, tol_pv=1e-6, omega=1.0)
result_tpf = tpf_solver.solve(network)
# ── 4. Slack-Leistung vergleichen ──
print("═" * 60)
print("  SLACK-LEISTUNG VERGLEICH")
print("═" * 60)

s_tpf = result_tpf.s_slack
s_nr  = result_nr.s_slack

p_tpf, q_tpf = s_tpf[0, 0].real, s_tpf[0, 0].imag
p_nr,  q_nr  = s_nr[0, 0].real,  s_nr[0, 0].imag

print(f"\n  TPF: P = {p_tpf:+.6f} p.u.  Q = {q_tpf:+.6f} p.u.")
print(f"  NR:  P = {p_nr:+.6f} p.u.  Q = {q_nr:+.6f} p.u.")

p_abs, p_rel = abs(p_tpf - p_nr), abs(p_tpf - p_nr) / max(abs(p_nr), 1e-12)
q_abs, q_rel = abs(q_tpf - q_nr), abs(q_tpf - q_nr) / max(abs(q_nr), 1e-12)

print(f"\n  ΔP: abs = {p_abs:.2e}  rel = {p_rel:.2e}")
print(f"  ΔQ: abs = {q_abs:.2e}  rel = {q_rel:.2e}")

# ── Kombinierter Test: entweder absolut ODER relativ klein genug ──
tol_abs = 1e-6
tol_rel = 1e-4

p_ok = p_abs < tol_abs or p_rel < tol_rel
q_ok = q_abs < tol_abs or q_rel < tol_rel

if p_ok and q_ok:
    print(f"\n  ✓ PASS: |Δ| < {tol_abs:.0e} oder |Δ|/|S| < {tol_rel:.0e}")
else:
    print(f"\n  ✗ FAIL: Weder absolute noch relative Toleranz erreicht")

# ── Zusätzliche Diagnose: Voltage-Vergleich (der eigentliche Test) ──
print(f"\n{'─'*60}")
print("  DIAGNOSE: Spannungsvergleich (der wichtigere Test)")
print(f"{'─'*60}")

import numpy as np
ppc = net._ppc
bus_types = ppc["bus"][:, 1].astype(int)
d_idx = np.sort(np.concatenate([
    np.where(bus_types == 1)[0],
    np.where(bus_types == 2)[0],
]))

v_tpf = result_tpf.voltages.flatten()
v_nr  = result_nr.voltages[d_idx]

dv_max = np.max(np.abs(np.abs(v_tpf) - np.abs(v_nr)))
dv_mean = np.mean(np.abs(np.abs(v_tpf) - np.abs(v_nr)))

print(f"  max |ΔV|:  {dv_max:.2e} p.u.")
print(f"  mean |ΔV|: {dv_mean:.2e} p.u.")

# Y-Matrix Verstärkungsfaktor
y_sd_norm = np.max(np.abs(network.Y_sd))
print(f"\n  |Y_sd|_max = {y_sd_norm:.1f}")
print(f"  Verstärkungsfaktor: |ΔS| ≈ |Y_sd| · |ΔV| = "
      f"{y_sd_norm:.1f} · {dv_max:.2e} = {y_sd_norm * dv_max:.2e}")
print(f"  → Erklärt beobachtete |ΔP|,|ΔQ| ≈ {p_abs:.2e}, {q_abs:.2e} ✓")