"""
Analytischer 2-Bus-Test
========================
Slack (Bus 0) --- Leitung z=0.1+j0.2 --- Last (Bus 1, S=0.5+j0.2 p.u.)

Bekannte analytische Lösung über quadratische Gleichung.
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tpf.core.network import NetworkData
from tpf.solvers.tpf_dense import TPFDenseSolver


def create_2bus_network():
    """Erstellt ein minimales 2-Bus-Netz manuell."""
    # Leitungsparameter
    z_line = 0.1 + 0.2j  # p.u.
    y_line = 1.0 / z_line

    # Admittanzmatrix (2×2 gesamt, 1×1 nach Partitionierung)
    # Bus 0 = Slack, Bus 1 = PQ (Last)
    #
    # Y = [y_line,  -y_line]
    #     [-y_line,  y_line]
    #
    # Y_dd = [y_line]  (1×1, nur Bus 1)
    # Y_ds = [-y_line] (1×1, Kopplung Bus 1 ↔ Bus 0)

    Y_dd = np.array([[y_line]], dtype=np.complex128)
    Y_ds = np.array([[-y_line]], dtype=np.complex128)
    v_s = np.array([1.0 + 0j], dtype=np.complex128)  # Slack = 1∠0°

    # Last am Bus 1: S_load = 0.5 + j0.2 p.u. (Verbraucherkonvention)
    s_nom = np.array([0.5 + 0.2j], dtype=np.complex128)

    return NetworkData(
        Y_dd=Y_dd,
        Y_ds=Y_ds,
        v_s=v_s,
        s_nom=s_nom,
        alpha_p=np.array([1.0]),
        alpha_i=np.array([0.0]),
        alpha_z=np.array([0.0]),
        n_buses=1,
        n_phases=1,
    )


def analytical_solution(z_line, s_load, v_slack=1.0):
    """
    Berechnet die analytische Lösung des 2-Bus-Problems.

    Aus: V_load = V_slack - z_line * I_load
    Mit: I_load = conj(S_load / V_load)

    Führt zu quadratischer Gleichung in |V|^2.
    """
    r = z_line.real
    x = z_line.imag
    p = s_load.real
    q = s_load.imag

    # |V|^4 - (|Vs|^2 - 2(rP + xQ))|V|^2 + (r^2+x^2)(P^2+Q^2) = 0
    # a·u² + b·u + c = 0, mit u = |V|²
    a = 1.0
    b = -(v_slack ** 2 - 2 * (r * p + x * q))
    c = (r ** 2 + x ** 2) * (p ** 2 + q ** 2)

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        print("KEINE LÖSUNG (Last zu hoch!)")
        return None

    u1 = (-b + np.sqrt(discriminant)) / (2 * a)  # High-voltage solution
    u2 = (-b - np.sqrt(discriminant)) / (2 * a)  # Low-voltage solution

    v_mag_hv = np.sqrt(u1)  # Operative Lösung
    print(f"  Analytisch |V| (High-Voltage): {v_mag_hv:.8f} p.u.")
    print(f"  Analytisch |V| (Low-Voltage):  {np.sqrt(u2):.8f} p.u.")

    return v_mag_hv


def main():
    print("=" * 50)
    print("  2-Bus Analytischer Test")
    print("=" * 50)

    z_line = 0.1 + 0.2j
    s_load = 0.5 + 0.2j

    # Analytische Lösung
    print("\n[Analytisch]")
    v_analytical = analytical_solution(z_line, s_load)

    # TPF-Lösung
    print("\n[TPF Dense]")
    network = create_2bus_network()
    solver = TPFDenseSolver(tol=1e-10, max_iter=100)
    result = solver.solve(network)

    v_tpf = np.abs(result.voltages.flatten()[0])
    print(f"  TPF |V|:          {v_tpf:.8f} p.u.")
    print(f"  Konvergiert:      {result.converged}")
    print(f"  Iterationen:      {result.iterations}")
    print(f"  Max Mismatch:     {result.max_mismatch:.2e}")

    # Vergleich
    error = abs(v_tpf - v_analytical)
    print(f"\n[Vergleich]")
    print(f"  |V| Fehler: {error:.2e} p.u.")
    print(f"  Status: {'PASS' if error < 1e-8 else 'FAIL'}")


if __name__ == "__main__":
    main()