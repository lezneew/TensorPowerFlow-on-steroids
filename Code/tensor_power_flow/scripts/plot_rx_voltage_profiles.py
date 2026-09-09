# tensor_power_flow/scripts/plot_rx_voltage_profiles.py
"""
Plots voltage profiles for 350 bus network at R/X = 0.1 and R/X = 10.
Creates side-by-side subplots for comparison, saved as PGF and PDF.

Aufruf:
    python -m scripts.plot_rx_voltage_profiles --pgf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import networkx as nx
import pandapower as pp
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
OUT_DEFAULT = Path(r"D:\Projects\TPF\TensorPowerFlow-on-steroids\Bachelor_tensorflow\figures")


def setup_mpl(use_pgf: bool):
    if use_pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        })
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
    })
    global plt
    import matplotlib.pyplot as plt  # noqa: E402
    globals()["USE_PGF"] = use_pgf


def build_graph_from_net(net: pp.pandapowerNet):
    """Build NetworkX graph from pandapower network with line lengths as edge weights."""
    G = nx.Graph()

    n_bus = len(net.bus)
    for i in range(n_bus):
        G.add_node(i)

    for _, line in net.line.iterrows():
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        length_km = line.get("length_km", 1.0)
        G.add_edge(from_bus, to_bus, weight=length_km)

    for _, trafo in net.trafo.iterrows():
        from_bus = int(trafo["hv_bus"])
        to_bus = int(trafo["lv_bus"])
        G.add_edge(from_bus, to_bus, weight=1.0)

    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    slack_idx = int(np.where(bus_types == 3)[0][0]) if np.any(bus_types == 3) else 0

    return G, slack_idx


def compute_slack_distances(G: nx.Graph, slack_idx: int, n_bus: int):
    """Compute shortest electrical distance (km) from slack bus to each bus."""
    distances = np.full(n_bus, np.inf)
    distances[slack_idx] = 0.0

    if slack_idx in G.nodes():
        try:
            lengths = nx.single_source_dijkstra_path_length(G, slack_idx, weight="weight")
            for bus, dist in lengths.items():
                distances[bus] = dist
        except nx.NetworkXError:
            pass

    distances = np.where(np.isinf(distances), 0.0, distances)
    return distances


def get_bus_types(net: pp.pandapowerNet):
    """Get bus type array from pandapower network (PPC convention)."""
    ppc = net._ppc
    return ppc["bus"][:, 1].astype(int)


def get_voltages(net: pp.pandapowerNet):
    """Get voltage magnitudes in p.u."""
    if hasattr(net, "res_bus") and "vm_pu" in net.res_bus.columns:
        return net.res_bus["vm_pu"].values
    return np.abs(net._ppc["bus"][:, 7])


def run_powerflow(net: pp.pandapowerNet):
    """Run Newton-Raphson power flow."""
    try:
        pp.runpp(net, verbose=False)
        return net.converged
    except Exception:
        return False


def plot_single_profile(ax, net, title, xlim=None, ylim=None):
    """Plot voltage profile on a single axis."""
    n_bus = len(net.bus)
    bus_types = get_bus_types(net)
    voltages = get_voltages(net)

    n_slack = np.sum(bus_types == 3)
    n_pv = np.sum(bus_types == 2)
    n_pq = np.sum(bus_types == 1)

    G, slack_idx = build_graph_from_net(net)
    slack_distances = compute_slack_distances(G, slack_idx, n_bus)

    for u, v in G.edges():
        x_coords = [slack_distances[u], slack_distances[v]]
        y_coords = [voltages[u], voltages[v]]
        ax.plot(x_coords, y_coords, color="gray", linewidth=0.8, alpha=0.5, zorder=1)

    pv_nodes = [i for i in range(n_bus) if bus_types[i] == 2]
    slack_nodes = [i for i in range(n_bus) if bus_types[i] == 3]
    pq_nodes = [i for i in range(n_bus) if bus_types[i] == 1]

    if pq_nodes:
        pq_x = [slack_distances[i] for i in pq_nodes]
        pq_y = [voltages[i] for i in pq_nodes]
        ax.scatter(pq_x, pq_y, c="black", s=1, marker="o", edgecolors="black",
                   linewidths=0.2, label=f"PQ ({n_pq})", zorder=3, alpha=0.8)

    if pv_nodes:
        pv_x = [slack_distances[i] for i in pv_nodes]
        pv_y = [voltages[i] for i in pv_nodes]
        ax.scatter(pv_x, pv_y, c="orange", s=100, marker="^", edgecolors="darkorange",
                   linewidths=1.0, label=f"PV ({n_pv})", zorder=4)

    if slack_nodes:
        slack_x = [slack_distances[i] for i in slack_nodes]
        slack_y = [voltages[i] for i in slack_nodes]
        ax.scatter(slack_x, slack_y, c="black", s=5, marker="s", edgecolors="black",
                   linewidths=0.5, label=f"Slack ({n_slack})", zorder=5)

    # ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="1.0 p.u.")
    # ax.axhline(y=1.05, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    # ax.axhline(y=0.95, color="red", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Distanz zum Slack (km)", fontsize=9)
    ax.set_ylabel("Spannung (p.u.)", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(left=-0.5)

    if ylim is not None:
        ax.set_ylim(ylim)


def main():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from tpf.generators.rx_sweep import build_rx_case, Z_REF

    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--pgf", action="store_true")
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    setup_mpl(args.pgf)

    print("Creating 350 bus networks...")

    case_01 = build_rx_case(nodes=350, pv_ratio=0.0, rx=0.1, mode="const_z",
                            z_abs=Z_REF, load_factor=2.0, seed=2350)
    case_10 = build_rx_case(nodes=350, pv_ratio=0.0, rx=10.0, mode="const_z",
                            z_abs=Z_REF, load_factor=2.0, seed=2350)

    print(f"  Network 1: R/X = 0.1, n_pv = {case_01.n_pv}")
    print(f"  Network 2: R/X = 10, n_pv = {case_10.n_pv}")

    print("Running power flow...")
    if not run_powerflow(case_01.net):
        raise RuntimeError("Power flow failed for R/X = 0.1")
    if not run_powerflow(case_10.net):
        raise RuntimeError("Power flow failed for R/X = 10")

    print("Computing axis limits...")
    G1, s1 = build_graph_from_net(case_01.net)
    G2, s2 = build_graph_from_net(case_10.net)
    d1 = compute_slack_distances(G1, s1, len(case_01.net.bus))
    d2 = compute_slack_distances(G2, s2, len(case_10.net.bus))
    v1 = get_voltages(case_01.net)
    v2 = get_voltages(case_10.net)

    x_min = min(d1.min(), d2.min())
    x_max = max(d1.max(), d2.max())
    x_margin = (x_max - x_min) * 0.05
    xlim = (x_min - x_margin, x_max + x_margin)

    v_min = min(v1.min(), v2.min())
    v_max = max(v1.max(), v2.max())
    v_margin = (v_max - v_min) * 0.1
    if v_margin == 0:
        v_margin = 0.02
    ylim = (v_min - v_margin, v_max + v_margin)

    print("Creating plot...")
    fig, ax = plt.subplots(1, 2, figsize=(5.91, 2.7), constrained_layout=True)

    plot_single_profile(ax[0], case_01.net, r"(a) $R/X = 0.1$", xlim=xlim, ylim=ylim)
    plot_single_profile(ax[1], case_10.net, r"(b) $R/X = 10$", xlim=xlim, ylim=ylim)

    for ext in ("pgf", "pdf"):
        try:
            fig.savefig(OUT / f"rx_voltage_profile_350.{ext}", bbox_inches="tight")
        except Exception as e:
            print(f"  savefig {ext}: {type(e).__name__}: {e}")
    print(f"  -> rx_voltage_profile_350.pgf/.pdf")
    plt.close(fig) if globals().get("USE_PGF") else plt.show()


if __name__ == "__main__":
    main()
