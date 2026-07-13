"""
Network Topology and Contraction Analysis Plotter
===================================================
Plots network topology with PV nodes highlighted and key matrices:
- Z_B: Bus impedance matrix (PV-PV submatrix)
- H: Sensitivity matrix ∂|V|²/∂Q
- J_G: Iteration matrix I + omega*H*D^-1 (spectral radius rho(J_G))

Color scheme:
- PQ buses: blue
- PV buses: red/orange  
- Slack: green

Usage:
    python scripts/plot_network_topology.py
    python scripts/plot_network_topology.py --suite salazar
    python scripts/plot_network_topology.py --suite salazar_scaling
"""

import numpy as np
import scipy.linalg as la
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
import pandapower as pp
import pandapower.plotting as pp_plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.generators.network_generator_salazar import (
    get_salazar_pv_networks,
    get_salazar_scaling_networks,
)
from tpf.generators.ieee_pegase_networks import (
    get_all_standard_networks
)


def generate_radial_coordinates(net):
    """Generate radial/tree coordinates for network without x/y data."""
    import networkx as nx
    
    n_bus = len(net.bus)
    G = nx.Graph()
    G.add_nodes_from(range(n_bus))
    
    for idx in net.line.index:
        from_bus = int(net.line.at[idx, 'from_bus'])
        to_bus = int(net.line.at[idx, 'to_bus'])
        G.add_edge(from_bus, to_bus)
    
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    slack_candidates = np.where(bus_types == 3)[0]
    slack_idx = int(slack_candidates[0]) if len(slack_candidates) > 0 else 0
    
    if not nx.is_connected(G) and n_bus > 1:
        slack_idx = 0
    
    x = np.zeros(n_bus)
    y = np.zeros(n_bus)
    
    if n_bus <= 1:
        return x, y
    
    try:
        if nx.is_connected(G):
            bfs_tree = nx.bfs_tree(G, source=slack_idx)
            bfs_edges = list(bfs_tree.edges())
            
            levels = {slack_idx: 0}
            queue = [slack_idx]
            while queue:
                node = queue.pop(0)
                for neighbor in G.neighbors(node):
                    if neighbor not in levels:
                        levels[neighbor] = levels[node] + 1
                        queue.append(neighbor)
            
            max_level = max(levels.values()) if levels else 1
            nodes_by_level = {}
            for node, level in levels.items():
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
            
            for level, nodes in nodes_by_level.items():
                n_nodes = len(nodes)
                y_base = -level * 5.0
                for i, node in enumerate(nodes):
                    if n_nodes == 1:
                        x[node] = 0
                    else:
                        x[node] = (i - (n_nodes - 1) / 2) * 5.0
                    y[node] = y_base
        else:
            angles = np.linspace(0, 2 * np.pi, n_bus, endpoint=False)
            for i in range(n_bus):
                x[i] = 10 * np.cos(angles[i])
                y[i] = 10 * np.sin(angles[i])
    except:
        angles = np.linspace(0, 2 * np.pi, n_bus, endpoint=False)
        for i in range(n_bus):
            x[i] = 10 * np.cos(angles[i])
            y[i] = 10 * np.sin(angles[i])
    
    return x, y


def compute_matrices(network, omega=1.0, delta_q=1e-5):
    """
    Compute Z_B, H, D, J_G matrices for contraction analysis.
    
    Returns:
        Z_B: Full bus impedance matrix
        Z_B_pv: PV-PV submatrix of Z_B
        H: Sensitivity matrix ∂|V|²/∂Q
        D: Thévenin diagonal matrix
        J_G: Iteration matrix I + omega*H*D^-1
        rho: Spectral radius of J_G
        eta: Contraction factor η = ||Z_B * diag(conj(s_nom))||_1 / v_min_pu²
    """
    if not network.has_pv or network.n_pv == 0:
        return None, None, None, None, None, 0.0, 0.0
    
    pv_idx = network.pv_indices
    n_pv = len(pv_idx)
    bphi = network.n_bus_phases
    
    try:
        Z_B = la.inv(network.Y_dd)
    except:
        return None, None, None, None, None, 0.0, 0.0
    
    Z_B_pv = Z_B[np.ix_(pv_idx, pv_idx)]
    
    K = -Z_B
    L = (K @ network.Y_ds @ network.v_s).reshape(-1, 1)
    
    X_kk = np.imag(Z_B[pv_idx, pv_idx])
    
    D_inv = np.diag(1.0 / (2.0 * np.where(np.abs(X_kk) > 1e-12, X_kk, 1e-12)))
    D = np.diag(2.0 * X_kk)
    
    s_base = network.s_nom.copy().reshape(-1, 1)
    
    def solve_inner_fpi(s_work, max_iter=200, tol=1e-10):
        V = np.ones((bphi, 1), dtype=np.complex128)
        S_conj = np.conj(s_work)
        for _ in range(max_iter):
            LAMBDA = S_conj * (1.0 / np.conj(V))
            V_new = K @ LAMBDA + L
            if np.max(np.abs(np.abs(V_new) - np.abs(V))) < tol:
                return np.abs(V_new[pv_idx, 0]) ** 2
            V = V_new
        return np.abs(V[pv_idx, 0]) ** 2
    
    v2_base = solve_inner_fpi(s_base)
    
    H = np.zeros((n_pv, n_pv))
    for j in range(n_pv):
        s_pert = s_base.copy()
        s_pert[pv_idx[j], 0] += 1j * delta_q
        v2_pert = solve_inner_fpi(s_pert)
        H[:, j] = (v2_pert - v2_base) / delta_q
    
    D_diag = np.diag(D_inv)
    H_D = H * D_diag[np.newaxis, :]
    J_G = np.eye(n_pv) + omega * H_D
    
    eigenvalues = la.eigvals(J_G)
    rho = float(np.max(np.abs(eigenvalues)))
    
    v_nom = float(np.abs(network.v_s[0])) if hasattr(network, 'v_s') and network.v_s is not None else 1.0
    v_min_pu = v_nom
    s_nom_vec = np.conj(network.s_nom.flatten())
    Z_B_s = Z_B * s_nom_vec[np.newaxis, :]
    eta = float(np.linalg.norm(Z_B_s, ord=1) / (v_min_pu ** 2))
    
    return Z_B, Z_B_pv, H, D, J_G, rho, eta


def plot_matrix_heatmap(M, title, ax, cmap='RdBu_r', log_scale=False, pv_indices=None, slack_index=None):
    """Plot a matrix as a heatmap with optional PV section markers."""
    if M is None:
        ax.text(0.5, 0.5, 'No PV buses', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return
    
    if np.iscomplexobj(M):
        M_plot = np.abs(M)
    else:
        M_plot = M
    
    n = M.shape[0]
    
    max_val = np.max(np.abs(M_plot))
    if max_val > 0:
        if log_scale:
            norm = SymLogNorm(linthresh=1e-10, vmin=-max_val, vmax=max_val)
            im = ax.imshow(M_plot, cmap=cmap, norm=norm, aspect='equal')
        else:
            im = ax.imshow(M_plot, cmap=cmap, vmin=0, vmax=max_val, aspect='equal')
    else:
        im = ax.imshow(M_plot, cmap=cmap, aspect='equal')
    
    if pv_indices is not None and len(pv_indices) > 0:
        pv_indices = sorted(pv_indices)
        min_pv = min(pv_indices)
        max_pv = max(pv_indices)
        
        for pv in pv_indices:
            ax.axvline(x=pv - 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.8)
            ax.axhline(y=pv - 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.8)
        
        rect = mpatches.Rectangle(
            (min_pv - 0.5, min_pv - 0.5),
            max_pv - min_pv + 1,
            max_pv - min_pv + 1,
            linewidth=3, edgecolor='red', facecolor='none',
            linestyle='-', alpha=0.9
        )
        ax.add_patch(rect)
    
    if slack_index is not None:
        ax.axvline(x=slack_index - 0.5, color='green', linewidth=2, linestyle='-', alpha=0.9)
        ax.axhline(y=slack_index - 0.5, color='green', linewidth=2, linestyle='-', alpha=0.9)
        
        rect_slack = mpatches.Rectangle(
            (slack_index - 0.5, slack_index - 0.5),
            1, 1,
            linewidth=3, edgecolor='green', facecolor='none',
            linestyle='-', alpha=0.9
        )
        ax.add_patch(rect_slack)
    
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|value|', fontsize=10)
    
    return im


def plot_network_analysis(name, net_constructor, output_dir, omega=1.0):
    """Create comprehensive plot: topology + Z_B + H + J_G"""
    print(f"  Processing: {name}")
    
    try:
        net = net_constructor()
    except Exception as e:
        print(f"    ✗ Error creating network: {e}")
        return
    
    try:
        network = build_network_from_pandapower(net, include_pv=True)
    except Exception as e:
        print(f"    ✗ Error building network: {e}")
        return
    
    if not network.has_pv or network.n_pv == 0:
        print(f"    ─ Skipping (no PV buses)")
        return
    
    pv_idx = list(network.pv_indices)
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    n_bus = len(bus_types)
    
    slack_idx = np.where(bus_types == 3)[0]
    slack_idx = int(slack_idx[0]) if len(slack_idx) > 0 else 0
    
    import sys
    print(f"    calling compute_matrices...", file=sys.stderr)
    sys.stderr.flush()
    Z_B, Z_B_pv, H, D, J_G, rho, eta = compute_matrices(network, omega)
    print(f"    done", file=sys.stderr)
    sys.stderr.flush()
    
    if J_G is None:
        print(f"    ─ Skipping (no PV)")
        return
    
    print(f"    creating figure...", file=sys.stderr)
    sys.stderr.flush()
    fig = plt.figure(figsize=(16, 16))
    print(f"    figure done", file=sys.stderr)
    sys.stderr.flush()
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)
    print(f"    gs done", file=sys.stderr)
    sys.stderr.flush()
    fig.suptitle(f'{name}\nomega={omega}, eta={eta:.4f}, rho(J_G)={rho:.4f}', fontsize=14, fontweight='bold')
    print(f"    suptitle done", file=sys.stderr)
    sys.stderr.flush()
    
    print(f"    subplots...", file=sys.stderr)
    sys.stderr.flush()
    axes = fig.subplots(2, 2)
    print(f"    subplots done", file=sys.stderr)
    sys.stderr.flush()
    ax_topology = axes[0, 0]
    ax_zb = axes[0, 1]
    ax_h = axes[1, 0]
    ax_jg = axes[1, 1]
    print(f"    axes assigned", file=sys.stderr)
    sys.stderr.flush()
    
    print(f"    getting coordinates...", file=sys.stderr)
    sys.stderr.flush()
    if 'x' in net.bus.columns and 'y' in net.bus.columns:
        x = net.bus['x'].values
        y = net.bus['y'].values
    else:
        x, y = generate_radial_coordinates(net)
    print(f"    coordinates done", file=sys.stderr)
    sys.stderr.flush()
    
    print(f"    plotting lines... n_lines={len(net.line.index)}", file=sys.stderr)
    sys.stderr.flush()
    
    lines_data = []
    for idx in net.line.index:
        from_bus = int(net.line.at[idx, 'from_bus'])
        to_bus = int(net.line.at[idx, 'to_bus'])
        lines_data.append((from_bus, to_bus))
    print(f"    lines_data extracted", file=sys.stderr)
    sys.stderr.flush()
    
    for i, (from_bus, to_bus) in enumerate(lines_data):
        ax_topology.plot([x[from_bus], x[to_bus]], [y[from_bus], y[to_bus]], 
                       'gray', linewidth=1.5, alpha=0.6, zorder=1)
        if i == 0:
            print(f"    first line done", file=sys.stderr)
            sys.stderr.flush()
    print(f"    all lines done", file=sys.stderr)
    sys.stderr.flush()
    
    print(f"    setting box aspect...", file=sys.stderr)
    sys.stderr.flush()
    ax_topology.set_box_aspect(1)
    print(f"    box aspect done", file=sys.stderr)
    sys.stderr.flush()
    
    print(f"    plotting buses... n_bus={n_bus}", file=sys.stderr)
    sys.stderr.flush()
    plotted_labels = {'slack': False, 'pv': False, 'pq': False}
    for i in range(n_bus):
        if i % 10 == 0:
            print(f"    bus {i}", file=sys.stderr)
            sys.stderr.flush()
        if i == slack_idx:
            color = 'green'
            size = 120
            label = 'Slack'
            plotted_labels['slack'] = True
        elif i in pv_idx:
            color = 'red'
            size = 100
            label = 'PV'
            if not plotted_labels['pv']:
                plotted_labels['pv'] = True
            else:
                label = None
        else:
            color = 'blue'
            size = 80
            label = 'PQ'
            if not plotted_labels['pq']:
                plotted_labels['pq'] = True
            else:
                label = None
        
        ax_topology.scatter(x[i], y[i], c=color, s=size, 
                           edgecolors='black', linewidths=1.5, 
                           zorder=5, label=label)
    print(f"    scatter done", file=sys.stderr)
    sys.stderr.flush()
    
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
              markersize=12, label='Slack'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
              markersize=10, label='PV'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
              markersize=8, label='PQ'),
    ]
    ax_topology.legend(handles=handles, loc='best', fontsize=9)
    print(f"    legend done", file=sys.stderr)
    sys.stderr.flush()
    ax_topology.set_xlabel('x [km]')
    ax_topology.set_ylabel('y [km]')
    print(f"    labels done", file=sys.stderr)
    sys.stderr.flush()
    
    ax_topology.set_title(f'Network Topology\n{n_bus} buses, {len(pv_idx)} PV', fontsize=12)
    ax_topology.grid(True, alpha=0.3)
    ax_topology.set_box_aspect(1)
    print(f"    topology done", file=sys.stderr)
    sys.stderr.flush()
    
    print(f"    plotting Z_B...", file=sys.stderr)
    sys.stderr.flush()
    plot_matrix_heatmap(Z_B, f'Z_B (full bus impedance)\n|Z_B| max={np.max(np.abs(Z_B)):.2e}', 
                       ax_zb, cmap='RdBu_r', log_scale=True, 
                       pv_indices=pv_idx, slack_index=slack_idx)
    ax_zb.set_xlabel('bus j')
    ax_zb.set_ylabel('bus i')
    
    plot_matrix_heatmap(H, f'H (Sensitivity d|V|^2/dQ)\n|H| max={np.max(np.abs(H)):.2e}', 
                       ax_h, cmap='RdBu_r', log_scale=True)
    ax_h.set_xlabel('PV bus j (Q perturbation)')
    ax_h.set_ylabel('PV bus i (|V|^2 response)')
    
    plot_matrix_heatmap(J_G, f'J_G = I + omega*H*D^-1\n|J_G| max={np.max(np.abs(J_G)):.2e}, rho={rho:.4f}', 
                       ax_jg, cmap='RdBu_r', log_scale=True)
    ax_jg.set_xlabel('PV bus j')
    ax_jg.set_ylabel('PV bus i')
    
    if rho < 1.0 and eta < 1.0:
        status_color = 'green'
        status_text = f'eta = {eta:.4f}, rho = {rho:.4f} -> FULL CONVERGENCE'
    elif rho < 1.0:
        status_color = 'orange'
        status_text = f'eta = {eta:.4f}, rho = {rho:.4f} -> OUTER CONVERGES (eta>=1, inner may not)'
    elif eta < 1.0:
        status_color = 'orange'
        status_text = f'eta = {eta:.4f}, rho = {rho:.4f} -> INNER CONVERGES (rho>=1, outer may not)'
    else:
        status_color = 'red'
        status_text = f'eta = {eta:.4f}, rho = {rho:.4f} -> MAY DIVERGE'
    
    fig.text(0.5, 0.02, status_text, ha='center', fontsize=12, 
            fontweight='bold', color=status_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{name}_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"    ✓ Saved: {save_path}")
    
    return {
        'name': name,
        'n_bus': n_bus,
        'n_pv': len(pv_idx),
        'eta': eta,
        'rho': rho,
        'saved_path': save_path
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot network topology and contraction matrices')
    parser.add_argument('--suite', choices=['salazar', 'salazar_pv', 'salazar_scaling', 'all', 'standard'],
                        default='salazar', help='Test suite to plot')
    parser.add_argument('--omega', type=float, default=1.0, help='Relaxation factor omega (default: 1.0)')
    parser.add_argument('--output', type=str, default='topology', help='Output directory (default: topology)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("NETWORK TOPOLOGY & CONTRACTION ANALYSIS PLOTTER")
    print("Plots: Topology + Z_B + H + J_G matrices")
    print("=" * 70)
    
    networks = {}
    if args.suite in ('salazar', 'all'):
        networks.update(get_salazar_pv_networks())
    if args.suite in ('salazar_pv', 'all'):
        networks.update(get_salazar_pv_networks())
    if args.suite in ('salazar_scaling', 'all'):
        networks.update(get_salazar_scaling_networks())
    if args.suite in ('standard', 'all'):
        networks.update(get_all_standard_networks())
    
    print(f"\n  Suite: '{args.suite}' - {len(networks)} networks")
    print(f"  omega = {args.omega}")
    print(f"  Output: {args.output}/\n")
    
    results = []
    for name, info in networks.items():
        result = plot_network_analysis(name, info['constructor'], args.output, omega=args.omega)
        if result:
            results.append(result)
    
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Networks processed: {len(results)}")
    
    if results:
        etas = [r['eta'] for r in results]
        rhos = [r['rho'] for r in results]
        print(f"  eta range: [{min(etas):.4f}, {max(etas):.4f}]")
        print(f"  rho(J_G) range: [{min(rhos):.4f}, {max(rhos):.4f}]")
        print(f"  Networks with eta < 1: {sum(1 for e in etas if e < 1.0)}")
        print(f"  Networks with rho < 1: {sum(1 for r in rhos if r < 1.0)}")
        print(f"  Networks with both eta < 1 and rho < 1: {sum(1 for r in results if r['eta'] < 1.0 and r['rho'] < 1.0)}")
        print(f"\n  Output saved to: {args.output}/")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()