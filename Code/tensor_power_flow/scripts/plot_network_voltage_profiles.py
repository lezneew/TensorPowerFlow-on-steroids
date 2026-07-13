# tensor_power_flow/scripts/plot_network_voltage_profiles.py
"""
Plot voltage profiles for all test networks.
Creates one PNG per network showing:
- X-axis: Slack distance (electrical km from slack bus)
- Y-axis: Voltage magnitude (p.u.)
- Nodes colored by type: PV (orange triangles), PQ (blue circles), Slack (green star)
- Edges shown as lines connecting buses (topology visible)
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from tpf.generators.network_generator_salazar import SALAZAR_TEST_NETWORKS, SALAZAR_SCALING_NETWORKS, SALAZAR_LOW_VM_NETWORKS
from tpf.generators.radial_network import TEST_NETWORKS
from setup_test_networks import create_4bus_1pv, create_33bus_with_dg, get_ieee30, get_ieee57


OUTPUT_DIR = "voltage_profiles"


def build_graph_from_net(net: pp.pandapowerNet) -> tuple[nx.Graph, int]:
    """
    Build NetworkX graph from pandapower network with line lengths as edge weights.
    Returns graph and slack bus index.
    """
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


def compute_slack_distances(G: nx.Graph, slack_idx: int, n_bus: int) -> np.ndarray:
    """
    Compute shortest electrical distance (km) from slack bus to each bus.
    Uses Dijkstra with line lengths as weights.
    """
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


def get_bus_types(net: pp.pandapowerNet) -> np.ndarray:
    """Get bus type array from pandapower network (PPC convention)."""
    ppc = net._ppc
    return ppc["bus"][:, 1].astype(int)


def get_voltages(net: pp.pandapowerNet) -> np.ndarray:
    """Get voltage magnitudes in p.u."""
    if hasattr(net, "res_bus") and "vm_pu" in net.res_bus.columns:
        return net.res_bus["vm_pu"].values
    return np.abs(net._ppc["bus"][:, 7])


def plot_network(net: pp.pandapowerNet, name: str, output_dir: str):
    """Plot single network voltage profile with slack distance on x-axis."""
    n_bus = len(net.bus)
    bus_types = get_bus_types(net)
    voltages = get_voltages(net)

    n_slack = np.sum(bus_types == 3)
    n_pv = np.sum(bus_types == 2)
    n_pq = np.sum(bus_types == 1)

    G, slack_idx = build_graph_from_net(net)

    if len(G.nodes()) == 0:
        print(f"  [{name}] No nodes, skipping")
        return False

    slack_distances = compute_slack_distances(G, slack_idx, n_bus)

    fig, ax = plt.subplots(figsize=(14, 8))

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
        ax.scatter(pq_x, pq_y, c="steelblue", s=60, marker="o", edgecolors="black",
                   linewidths=0.5, label=f"PQ ({n_pq})", zorder=3, alpha=0.8)

    if pv_nodes:
        pv_x = [slack_distances[i] for i in pv_nodes]
        pv_y = [voltages[i] for i in pv_nodes]
        ax.scatter(pv_x, pv_y, c="orange", s=100, marker="^", edgecolors="darkorange",
                   linewidths=1.0, label=f"PV ({n_pv})", zorder=4)

    if slack_nodes:
        slack_x = [slack_distances[i] for i in slack_nodes]
        slack_y = [voltages[i] for i in slack_nodes]
        ax.scatter(slack_x, slack_y, c="green", s=200, marker="*", edgecolors="black",
                   linewidths=0.5, label=f"Slack ({n_slack})", zorder=5)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="1.0 p.u.")
    ax.axhline(y=1.05, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.95, color="red", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Distance from Slack Bus (km)", fontsize=12)
    ax.set_ylabel("Voltage (p.u.)", fontsize=12)
    ax.set_title(f"{name}\n{n_bus} buses | Slack: {n_slack} | PV: {n_pv} | PQ: {n_pq}",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.5)

    v_min, v_max = voltages.min(), voltages.max()
    v_margin = (v_max - v_min) * 0.1
    ax.set_ylim(v_min - v_margin, v_max + v_margin)

    plt.tight_layout()

    safe_name = name.replace(" ", "_").replace("/", "_")
    filepath = os.path.join(output_dir, f"{safe_name}_voltage.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  [{name}] Saved: {filepath}")
    return True


def run_powerflow(net: pp.pandapowerNet) -> bool:
    """Run Newton-Raphson power flow."""
    try:
        pp.runpp(net, verbose=False)
        return net.converged
    except Exception:
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot voltage profiles for test networks")
    parser.add_argument(
        "--suite",
        choices=["salazar_scaling", "salazar_low_vm", "salazar_all"],
        default="salazar_scaling",
        help="Network suite to plot: salazar_scaling (default), salazar_low_vm (low voltage), salazar_all (both)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: voltage_profiles)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    all_networks = {}

    if args.suite in ["salazar_scaling", "salazar_all"]:
        for name, info in SALAZAR_SCALING_NETWORKS.items():
            all_networks[f"salazar_{name}"] = info

    if args.suite in ["salazar_low_vm", "salazar_all"]:
        for name, info in SALAZAR_LOW_VM_NETWORKS.items():
            all_networks[f"salazar_{name}"] = info

    # for name, info in SALAZAR_TEST_NETWORKS.items():
    #     all_networks[f"salazar_{name}"] = info
    #
    # for name, info in TEST_NETWORKS.items():
    #     all_networks[f"radial_{name}"] = info
    #
    # setup_networks = {
    #     "setup_4bus_1pv": {
    #         "constructor": create_4bus_1pv,
    #         "description": "Setup: 4bus 1pv",
    #         "n_pv": 1,
    #     },
    #     "setup_33bus_2dg": {
    #         "constructor": create_33bus_with_dg,
    #         "description": "Setup: 33bus 2dg",
    #         "n_pv": 2,
    #     },
    #     "setup_ieee30": {
    #         "constructor": get_ieee30,
    #         "description": "Setup: IEEE30",
    #         "n_pv": 5,
    #     },
    #     "setup_ieee57": {
    #         "constructor": get_ieee57,
    #         "description": "Setup: IEEE57",
    #         "n_pv": 6,
    #     },
    # }
    # for name, info in setup_networks.items():
    #     all_networks[name] = info

    print(f"Suite: {args.suite}")
    print(f"Total networks to process: {len(all_networks)}")

    converged = 0
    skipped = 0
    failed = 0

    for name, info in all_networks.items():
        print(f"\nProcessing: {name}")

        try:
            net = info["constructor"]()
        except Exception as e:
            print(f"  [{name}] Failed to create network: {e}")
            failed += 1
            continue

        if not run_powerflow(net):
            print(f"  [{name}] Power flow did not converge, skipping")
            skipped += 1
            continue

        try:
            if plot_network(net, name, output_dir):
                converged += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [{name}] Plotting failed: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {converged} plots saved, {skipped} skipped (no converge), {failed} failed")
    print(f"Output directory: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()