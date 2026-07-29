"""
Test Load Factors lambda for Tensor Power Flow
==========================================

Tests different load factors lambda to analyze convergence behavior and compare
TPF (Method A) vs Newton-Raphson solvers.

Usage:
    python scripts/test_load_factors.py
    python scripts/test_load_factors.py --network case100_5pv
    python scripts/test_load_factors.py --suite salazar_scaling --show-plot
    python scripts/test_load_factors.py --f3-study --show-plot
    python scripts/test_load_factors.py --save-json results.json
"""

import numpy as np
import sys
import os
import warnings
import argparse
import json
from typing import Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.radial_network import get_quick_test_networks
from tpf.generators.network_generator_salazar import (
    create_salazar_network,
    get_salazar_scaling_networks,
    get_salazar_all_networks,
    get_salazar_pv_networks,
    get_salazar_small_pv_networks,
    get_salazar_large_pv_networks,
    get_salazar_paper_networks,
    get_salazar_low_rx05_networks,
    get_salazar_low_rx10_networks,
)

VALID_SUITES = [
    "quick",
    "salazar_scaling",
    "salazar_all",
    "salazar_pv",
    "salazar_small_pv",
    "salazar_large_pv",
    "salazar_paper",
]


def get_suite_networks(suite_name: str) -> dict:
    """Get networks for a single suite name."""
    suites = {
        "quick": get_quick_test_networks,
        "salazar_scaling": get_salazar_scaling_networks,
        "salazar_all": get_salazar_all_networks,
        "salazar_pv": get_salazar_pv_networks,
        "salazar_small_pv": get_salazar_small_pv_networks,
        "salazar_large_pv": get_salazar_large_pv_networks,
        "salazar_paper": get_salazar_paper_networks,
    }
    if suite_name not in suites:
        raise ValueError(f"Unknown suite: '{suite_name}'. Valid: {VALID_SUITES}")
    return suites[suite_name]()


def build_s_batch(network, lambdas: np.ndarray) -> np.ndarray:
    """
    Build s_batch where each column is lambda * s_nom.

    For PV nodes: keep P scaled, Q must be 0 (solver finds it)
    """
    s_nom = network.s_nom.reshape(-1, 1)
    s_batch = s_nom * lambdas.reshape(1, -1)

    if network.has_pv:
        pv = network.pv_indices
        s_batch[pv, :] = s_batch[pv, :].real + 0j

    return s_batch


def build_nr_s_batch(net, lambdas: np.ndarray) -> tuple:
    """
    Build s_batch for NR solver (PQ loads only, in MW/Mvar).
    Returns (p_load, q_load) profiles.
    """
    p_load = net.load["p_mw"].values
    q_load = net.load["q_mvar"].values

    n_load = len(p_load)
    tau = len(lambdas)

    p_profile = p_load.reshape(-1, 1) * lambdas.reshape(1, -1)
    q_profile = q_load.reshape(-1, 1) * lambdas.reshape(1, -1)

    return p_profile, q_profile


def test_tpf_lambda_sweep(
    network,
    lambdas: np.ndarray,
    max_inner: int = 50,
    max_outer: int = 30,
    tol: float = 1e-6,
    tol_pv: float = 1e-6,
    omega: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Test TPF solver across lambda sweep."""
    s_batch = build_s_batch(network, lambdas)

    solver = TPFDensePVMethodA(
        tol=tol,
        max_iter_inner=max_inner,
        max_iter_outer=max_outer,
        tol_pv=tol_pv,
        omega=omega,
        enforce_q_lims=False,
        cold_start=False,
        adaptive_inner=False,
    )

    if verbose:
        print(f"    Running TPF timeseries solve for {len(lambdas)} lambdas...")

    result = solver.solve_timeseries(network, s_batch, verbose=verbose)

    info = solver.pv_info

    return {
        "converged_per_scenario": info.converged_per_scenario,
        "outer_iterations_per_scenario": info.outer_iterations_per_scenario,
        "inner_iterations_per_scenario": info.inner_iterations_per_scenario,
        "pv_v_error_per_scenario": info.pv_v_error_per_scenario,
        "elapsed_time_s": result.elapsed_time_s,
        "converged": result.converged,
        "n_converged": info.n_converged_scenarios,
        "n_scenarios": info.n_scenarios,
    }


def test_nr_lambda_sweep(
    net,
    lambdas: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
) -> dict:
    """Test Newton-Raphson solver across lambda sweep."""
    p_profile, q_profile = build_nr_s_batch(net, lambdas)

    solver = PandapowerNRSolver(tol=tol, max_iter=max_iter)

    if verbose:
        print(f"    Running NR timeseries solve for {len(lambdas)} lambdas...")

    result = solver.solve_timeseries(net, p_profile, q_profile, verbose=verbose)

    return {
        "converged_per_scenario": np.array([result.converged] * len(lambdas)),
        "iterations_per_scenario": np.array([result.iterations] * len(lambdas)),
        "elapsed_time_s": result.elapsed_time_s,
        "converged": result.converged,
    }


def print_lambda_results(
    lambdas: np.ndarray,
    tpf_results: dict,
    nr_results: dict,
    network_name: str,
):
    """Print detailed per-lambda results table."""
    print(f"\n{'='*140}")
    print(f"  LOAD FACTOR lambda RESULTS: {network_name}")
    print(f"{'='*140}")

    header = (
        f"  {'lambda':>6} "
        f"{'TPF Conv':>9} "
        f"{'TPF Out':>8} "
        f"{'TPF In':>7} "
        f"{'TPF Err':>12} "
        f"{'NR Conv':>8} "
        f"{'NR It':>7} "
    )
    print(header)
    print(f"  {'-'*138}")

    tpf_conv = tpf_results["converged_per_scenario"]
    tpf_outer = tpf_results["outer_iterations_per_scenario"]
    tpf_inner = tpf_results["inner_iterations_per_scenario"]
    tpf_err = tpf_results["pv_v_error_per_scenario"]

    nr_conv = nr_results["converged_per_scenario"]
    nr_iters = nr_results["iterations_per_scenario"]

    for i, lam in enumerate(lambdas):
        tpf_c = "Yes" if tpf_conv[i] else "No "
        tpf_o = f"{tpf_outer[i]:d}"
        tpf_i = f"{tpf_inner[i]:d}" if tpf_outer[i] > 0 else "-"
        tpf_e = f"{tpf_err[i]:.2e}" if np.isfinite(tpf_err[i]) else "inf"

        nr_c = "Yes" if nr_conv[i] else "No "
        nr_i = f"{nr_iters[i]:d}" if nr_conv[i] else "-"

        print(
            f"  {lam:>6.2f} "
            f"{tpf_c:>9} "
            f"{tpf_o:>8} "
            f"{tpf_i:>7} "
            f"{tpf_e:>12} "
            f"{nr_c:>8} "
            f"{nr_i:>7} "
        )

    print(f"  {'-'*138}")

    tpf_n_conv = np.sum(tpf_conv)
    nr_n_conv = np.sum(nr_conv)
    tpf_time = tpf_results["elapsed_time_s"] * 1000
    nr_time = nr_results["elapsed_time_s"] * 1000

    print(
        f"  Summary: TPF {tpf_n_conv}/{len(lambdas)} converged ({100*tpf_n_conv/len(lambdas):.1f}%) "
        f"in {tpf_time:.1f}ms | "
        f"NR {nr_n_conv}/{len(lambdas)} converged ({100*nr_n_conv/len(lambdas):.1f}%) "
        f"in {nr_time:.1f}ms"
    )

    tpf_max_lambda = lambdas[tpf_conv][-1] if np.any(tpf_conv) else 0.0
    nr_max_lambda = lambdas[nr_conv][-1] if np.any(nr_conv) else 0.0
    print(f"  Max lambda converged: TPF={tpf_max_lambda:.2f}, NR={nr_max_lambda:.2f}")
    print(f"{'='*140}\n")


def run_f3_study(
    base_nodes: int = 100,
    pv_ratios: list = None,
    lambdas: np.ndarray = None,
    seed: int = 42,
    show_plot: bool = False,
    verbose: bool = False,
) -> dict:
    """
    F3 Study: Vary lambda and n_PV together.

    This is exactly what the expose's F3 asks for:
    - Grid search: lambda ∈ [0.3, 2.0], n_PV ratios ∈ [0%, 5%, 10%, 20%, 30%]
    - Store outer iterations for each combination
    """
    if pv_ratios is None:
        pv_ratios = [0.0, 0.05, 0.10, 0.20, 0.30]
    if lambdas is None:
        lambdas = np.linspace(0.3, 2.0, 18)

    results = np.full((len(pv_ratios), len(lambdas)), np.nan)
    converged = np.full((len(pv_ratios), len(lambdas)), False, dtype=bool)

    if verbose:
        print(f"\n{'#'*80}")
        print(f"  F3 STUDY: lambda × n_PV ratio")
        print(f"  Nodes: {base_nodes}, lambda range: [{lambdas[0]:.2f}, {lambdas[-1]:.2f}], PV ratios: {pv_ratios}")
        print(f"{'#'*80}\n")

    for r_idx, ratio in enumerate(pv_ratios):
        n_pv = int(round(base_nodes * ratio))
        if verbose:
            print(f"  Testing n_PV={n_pv} ({ratio*100:.0f}%)...")

        net = create_salazar_network(nodes=base_nodes, n_pv=n_pv, seed=seed)
        network = build_network_from_pandapower(net, include_pv=True)

        s_batch = build_s_batch(network, lambdas)

        solver = TPFDensePVMethodA(
            tol=1e-6,
            max_iter_inner=50,
            max_iter_outer=50,
            tol_pv=1e-6,
            omega=1.0,
            enforce_q_lims=False,
            cold_start=False,
            adaptive_inner=False,
        )

        result = solver.solve_timeseries(network, s_batch, verbose=False)
        info = solver.pv_info

        results[r_idx, :] = info.outer_iterations_per_scenario
        converged[r_idx, :] = info.converged_per_scenario

        if verbose:
            n_conv = np.sum(info.converged_per_scenario)
            print(f"    Converged: {n_conv}/{len(lambdas)}, lambda_max={lambdas[info.converged_per_scenario][-1] if n_conv > 0 else 0:.2f}")

    if show_plot:
        plot_f3_heatmap(lambdas, pv_ratios, results, converged)

    return {
        "lambdas": lambdas,
        "pv_ratios": pv_ratios,
        "results": results,
        "converged": converged,
    }


def run_rx_ratio_study(
    base_nodes: int = 100,
    rx_ratios: list = None,
    lambdas: np.ndarray = None,
    seed: int = 42,
    show_plot: bool = False,
    verbose: bool = False,
) -> dict:
    """
    R/X Ratio Study: Vary lambda and R/X together (4.2).

    Grid search: lambda ∈ [0.3, 2.0], R/X ∈ {0.5, 1.0, 2.0}
    """
    if rx_ratios is None:
        rx_ratios = [0.5, 1.0, 2.0]
    if lambdas is None:
        lambdas = np.linspace(0.3, 2.0, 18)

    results = np.full((len(rx_ratios), len(lambdas)), np.nan)
    converged = np.full((len(rx_ratios), len(lambdas)), False, dtype=bool)

    if verbose:
        print(f"\n{'#'*80}")
        print(f"  RX STUDY: lambda x R/X ratio")
        print(f"  Nodes: {base_nodes}, lambda range: [{lambdas[0]:.2f}, {lambdas[-1]:.2f}], R/X ratios: {rx_ratios}")
        print(f"{'#'*80}\n")

    for r_idx, rx in enumerate(rx_ratios):
        if verbose:
            print(f"  Testing R/X={rx}...")

        net = create_salazar_network(nodes=base_nodes, n_pv=5, rx_ratio=rx, seed=seed)
        network = build_network_from_pandapower(net, include_pv=True)

        s_batch = build_s_batch(network, lambdas)

        solver = TPFDensePVMethodA(
            tol=1e-6,
            max_iter_inner=50,
            max_iter_outer=50,
            tol_pv=1e-6,
            omega=1.0,
            enforce_q_lims=False,
            cold_start=False,
            adaptive_inner=False,
        )

        result = solver.solve_timeseries(network, s_batch, verbose=False)
        info = solver.pv_info

        results[r_idx, :] = info.outer_iterations_per_scenario
        converged[r_idx, :] = info.converged_per_scenario

        if verbose:
            n_conv = np.sum(info.converged_per_scenario)
            print(f"    Converged: {n_conv}/{len(lambdas)}, lambda_max={lambdas[info.converged_per_scenario][-1] if n_conv > 0 else 0:.2f}")

    if show_plot:
        plot_rx_heatmap(lambdas, rx_ratios, results, converged)

    return {
        "lambdas": lambdas,
        "rx_ratios": rx_ratios,
        "results": results,
        "converged": converged,
    }


def plot_rx_heatmap(
    lambdas: np.ndarray,
    rx_ratios: list,
    results: np.ndarray,
    converged: np.ndarray,
    save_path: str = None,
):
    """Plot R/X study results as heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    masked_results = np.ma.masked_where(~converged, results)

    ax1 = axes[0]
    im1 = ax1.imshow(masked_results, aspect="auto", origin="lower", cmap="viridis_r")
    ax1.set_xlabel("Load factor lambda", fontsize=11)
    ax1.set_ylabel("R/X ratio", fontsize=11)
    ax1.set_title("Outer Iterations (converged only)", fontsize=12)
    ax1.set_xticks(np.linspace(0, len(lambdas)-1, 5).astype(int))
    ax1.set_xticklabels([f"{lambdas[int(i)]:.1f}" for i in np.linspace(0, len(lambdas)-1, 5)])
    ax1.set_yticks(range(len(rx_ratios)))
    ax1.set_yticklabels([f"{r:.1f}" for r in rx_ratios])
    plt.colorbar(im1, ax=ax1, label="Outer iterations")

    ax2 = axes[1]
    im2 = ax2.imshow(converged.astype(int), aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xlabel("Load factor lambda", fontsize=11)
    ax2.set_ylabel("R/X ratio", fontsize=11)
    ax2.set_title("Convergence (green=conv, red=div)", fontsize=12)
    ax2.set_xticks(np.linspace(0, len(lambdas)-1, 5).astype(int))
    ax2.set_xticklabels([f"{lambdas[int(i)]:.1f}" for i in np.linspace(0, len(lambdas)-1, 5)])
    ax2.set_yticks(range(len(rx_ratios)))
    ax2.set_yticklabels([f"{r:.1f}" for r in rx_ratios])
    plt.colorbar(im2, ax=ax2, label="Converged")

    fig.suptitle("R/X Study: Load Factor lambda vs R/X Ratio", fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_f3_heatmap(
    lambdas: np.ndarray,
    pv_ratios: list,
    results: np.ndarray,
    converged: np.ndarray,
    save_path: str = None,
):
    """Plot F3 study results as heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    masked_results = np.ma.masked_where(~converged, results)

    ax1 = axes[0]
    im1 = ax1.imshow(masked_results, aspect="auto", origin="lower", cmap="viridis_r")
    ax1.set_xlabel("lambda index", fontsize=11)
    ax1.set_ylabel("PV ratio index", fontsize=11)
    ax1.set_title("Outer Iterations (converged only)", fontsize=12)
    ax1.set_xticks(np.linspace(0, len(lambdas)-1, 5).astype(int))
    ax1.set_xticklabels([f"{lambdas[int(i)]:.1f}" for i in np.linspace(0, len(lambdas)-1, 5)])
    ax1.set_yticks(range(len(pv_ratios)))
    ax1.set_yticklabels([f"{r*100:.0f}%" for r in pv_ratios])
    plt.colorbar(im1, ax=ax1, label="Outer iterations")

    ax2 = axes[1]
    im2 = ax2.imshow(converged.astype(int), aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xlabel("lambda index", fontsize=11)
    ax2.set_ylabel("PV ratio index", fontsize=11)
    ax2.set_title("Convergence (green=conv, red=div)", fontsize=12)
    ax2.set_xticks(np.linspace(0, len(lambdas)-1, 5).astype(int))
    ax2.set_xticklabels([f"{lambdas[int(i)]:.1f}" for i in np.linspace(0, len(lambdas)-1, 5)])
    ax2.set_yticks(range(len(pv_ratios)))
    ax2.set_yticklabels([f"{r*100:.0f}%" for r in pv_ratios])
    plt.colorbar(im2, ax=ax2, label="Converged")

    fig.suptitle("F3 Study: Load Factor lambda vs PV Ratio", fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_f3_heatmap_enhanced(
    lambdas: np.ndarray,
    pv_ratios: list,
    outer_iters: np.ndarray,
    inner_iters: np.ndarray,
    converged: np.ndarray,
    save_path: str = None,
):
    """
    Enhanced F3 heatmap (4.1) with 3 subplots:
    - Outer iterations, inner iterations, convergence
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    masked_outer = np.ma.masked_where(~converged, outer_iters)
    masked_inner = np.ma.masked_where(~converged, inner_iters)

    ax1 = axes[0]
    im1 = ax1.imshow(masked_outer, aspect="auto", origin="lower", cmap="Blues")
    ax1.set_xlabel("Load factor lambda", fontsize=11)
    ax1.set_ylabel("PV ratio", fontsize=11)
    ax1.set_title("(a) Outer Iterations", fontsize=12, fontweight="bold")
    ax1.set_xticks(np.linspace(0, len(lambdas)-1, 6).astype(int))
    ax1.set_xticklabels([f"{lambdas[int(i)]:.1f}" for i in np.linspace(0, len(lambdas)-1, 6)])
    ax1.set_yticks(range(len(pv_ratios)))
    ax1.set_yticklabels([f"{r*100:.0f}%" for r in pv_ratios])
    for i in range(len(pv_ratios)):
        for j in range(len(lambdas)):
            if converged[i, j]:
                ax1.text(j, i, f"{int(outer_iters[i, j])}", ha="center", va="center",
                        fontsize=7, color="white" if outer_iters[i, j] > 15 else "black")
    plt.colorbar(im1, ax=ax1, label="Outer iterations")

    ax2 = axes[1]
    im2 = ax2.imshow(masked_inner, aspect="auto", origin="lower", cmap="Oranges")
    ax2.set_xlabel("Load factor lambda", fontsize=11)
    ax2.set_ylabel("PV ratio", fontsize=11)
    ax2.set_title("(b) Total Inner Iterations", fontsize=12, fontweight="bold")
    ax2.set_xticks(np.linspace(0, len(lambdas)-1, 6).astype(int))
    ax2.set_xticklabels([f"{lambdas[int(i)]:.1f}" for i in np.linspace(0, len(lambdas)-1, 6)])
    ax2.set_yticks(range(len(pv_ratios)))
    ax2.set_yticklabels([f"{r*100:.0f}%" for r in pv_ratios])
    for i in range(len(pv_ratios)):
        for j in range(len(lambdas)):
            if converged[i, j]:
                ax2.text(j, i, f"{int(inner_iters[i, j])}", ha="center", va="center",
                        fontsize=7, color="white" if inner_iters[i, j] > 150 else "black")
    plt.colorbar(im2, ax=ax2, label="Inner iterations")

    ax3 = axes[2]
    im3 = ax3.imshow(converged.astype(int), aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax3.set_xlabel("Load factor lambda", fontsize=11)
    ax3.set_ylabel("PV ratio", fontsize=11)
    ax3.set_title("(c) Convergence Map", fontsize=12, fontweight="bold")
    ax3.set_xticks(np.linspace(0, len(lambdas)-1, 6).astype(int))
    ax3.set_xticklabels([f"{lambdas[int(i)]:.1f}" for i in np.linspace(0, len(lambdas)-1, 6)])
    ax3.set_yticks(range(len(pv_ratios)))
    ax3.set_yticklabels([f"{r*100:.0f}%" for r in pv_ratios])
    plt.colorbar(im3, ax=ax3, label="Converged (1/0)")

    conv_rate = np.sum(converged) / converged.size * 100
    fig.suptitle(f"(4.1) Enhanced F3 Heatmap: lambda x PV Ratio | Conv. Rate: {conv_rate:.1f}%",
                 fontsize=14, fontweight="bold", y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_lambda_sweep(
    lambdas: np.ndarray,
    tpf_results: dict,
    nr_results: dict,
    network_name: str,
    save_path: str = None,
):
    """Plot lambda sweep results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    tpf_conv = tpf_results["converged_per_scenario"]
    tpf_outer = tpf_results["outer_iterations_per_scenario"]
    tpf_err = tpf_results["pv_v_error_per_scenario"]
    nr_conv = nr_results["converged_per_scenario"]
    nr_iters = nr_results["iterations_per_scenario"]

    ax1 = axes[0, 0]
    ax1.plot(lambdas, tpf_conv.astype(int), "bo-", markersize=8, label="TPF", linewidth=2)
    ax1.plot(lambdas, nr_conv.astype(int), "r^--", markersize=8, label="NR", linewidth=2)
    ax1.set_xlabel("Load factor lambda", fontsize=11)
    ax1.set_ylabel("Converged (1=Yes, 0=No)", fontsize=11)
    ax1.set_title("(a) Convergence vs lambda", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["No", "Yes"])

    ax2 = axes[0, 1]
    mask = tpf_conv
    ax2.plot(lambdas[mask], tpf_outer[mask], "bo-", markersize=8, label="TPF outer", linewidth=2)
    ax2.plot(lambdas, nr_iters, "r^--", markersize=8, label="NR iterations", linewidth=2)
    ax2.set_xlabel("Load factor lambda", fontsize=11)
    ax2.set_ylabel("Iterations", fontsize=11)
    ax2.set_title("(b) Iterations vs lambda", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    mask = tpf_conv & np.isfinite(tpf_err)
    ax3.semilogy(lambdas[mask], tpf_err[mask], "bo-", markersize=8, label="TPF error", linewidth=2)
    ax3.axhline(y=1e-6, color="g", linestyle="--", label="tol=1e-6")
    ax3.set_xlabel("Load factor lambda", fontsize=11)
    ax3.set_ylabel("PV Voltage Error (log scale)", fontsize=11)
    ax3.set_title("(c) Error vs lambda", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    tpf_time = tpf_results["elapsed_time_s"] * 1000
    nr_time = nr_results["elapsed_time_s"] * 1000
    x = np.arange(len(lambdas))
    width = 0.35
    ax4.bar(x - width/2, [tpf_time] * len(lambdas), width, label="TPF", color="blue", alpha=0.7)
    ax4.bar(x + width/2, [nr_time] * len(lambdas), width, label="NR", color="red", alpha=0.7)
    ax4.set_xlabel("lambda index", fontsize=11)
    ax4.set_ylabel("Time (ms)", fontsize=11)
    ax4.set_title("(d) Execution Time", fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Load Factor lambda Sweep: {network_name}", fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_iterations_twin_axis(
    lambdas: np.ndarray,
    tpf_results: dict,
    nr_results: dict,
    network_name: str,
    save_path: str = None,
):
    """
    Enhanced iterations plot (1.1) with twin axis.
    - Left y-axis: outer iterations
    - Right y-axis: total inner iterations
    - Vertical dashed line at lambda_crit (first divergence)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    tpf_conv = tpf_results["converged_per_scenario"]
    tpf_outer = tpf_results["outer_iterations_per_scenario"]
    tpf_inner = tpf_results["inner_iterations_per_scenario"]
    nr_iters = nr_results["iterations_per_scenario"]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color_tpf_outer = "tab:blue"
    color_tpf_inner = "tab:orange"
    color_nr = "tab:red"

    mask = tpf_conv
    conv_lambdas = lambdas[mask]

    lambda_crit = conv_lambdas[-1] if len(conv_lambdas) > 0 else None

    ax1.set_xlabel("Load factor lambda", fontsize=12)
    ax1.set_ylabel("Outer iterations", color=color_tpf_outer, fontsize=12)
    line1 = ax1.plot(lambdas[mask], tpf_outer[mask], "o-", color=color_tpf_outer,
                     markersize=10, label="TPF outer iterations", linewidth=2.5)
    ax1.tick_params(axis="y", labelcolor=color_tpf_outer)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Total inner iterations", color=color_tpf_inner, fontsize=12)
    line2 = ax2.plot(lambdas[mask], tpf_inner[mask], "s-", color=color_tpf_inner,
                     markersize=8, label="TPF inner iterations", linewidth=2, alpha=0.8)
    ax2.tick_params(axis="y", labelcolor=color_tpf_inner)

    line3 = ax1.plot(lambdas, nr_iters, "^--", color=color_nr,
                     markersize=8, label="NR iterations", linewidth=2, alpha=0.7)

    if lambda_crit is not None:
        ax1.axvline(x=lambda_crit, color="purple", linestyle="--", linewidth=2,
                    label=f"lambda_crit = {lambda_crit:.2f}")
        ax1.annotate("DIVERGENCE", xy=(lambda_crit, ax1.get_ylim()[1]*0.9),
                    xytext=(lambda_crit + 0.1, ax1.get_ylim()[1]*0.85),
                    fontsize=10, color="purple", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="purple"))

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    if lambda_crit is not None:
        labels.append(f"lambda_crit = {lambda_crit:.2f}")
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    ax1.set_title(f"(1.1) Iterations vs Load Factor lambda: {network_name}", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


def test_single_network(
    net_name: str,
    lambdas: np.ndarray,
    max_outer: int = 30,
    show_plot: bool = False,
    verbose: bool = False,
) -> dict:
    """Test a single network with lambda sweep."""
    all_suites = [
        get_suite_networks("quick"),
        get_suite_networks("salazar_scaling"),
        get_suite_networks("salazar_pv"),
        get_suite_networks("salazar_all"),
    ]

    net_info = None
    for suite in all_suites:
        if net_name in suite:
            net_info = suite[net_name]
            break

    if net_info is None:
        raise ValueError(f"Unknown network: {net_name}")

    net_constructor = net_info["constructor"]
    net = net_constructor()
    network = build_network_from_pandapower(net, include_pv=True)

    if verbose:
        print(f"\n  Testing {net_name}: {network.n_buses} buses, {network.n_pv} PV nodes")

    tpf_results = test_tpf_lambda_sweep(
        network, lambdas, max_outer=max_outer, verbose=verbose
    )
    nr_results = test_nr_lambda_sweep(net, lambdas, verbose=verbose)

    print_lambda_results(lambdas, tpf_results, nr_results, net_name)

    if show_plot:
        plot_lambda_sweep(lambdas, tpf_results, nr_results, net_name)

    return {
        "network_name": net_name,
        "n_buses": network.n_buses,
        "n_pv": network.n_pv,
        "lambdas": lambdas.tolist(),
        "tpf": tpf_results,
        "nr": nr_results,
    }


def test_suite(
    suite_name: str,
    lambdas: np.ndarray,
    max_outer: int = 30,
    show_plot: bool = False,
    verbose: bool = False,
) -> dict:
    """Test all networks in a suite with lambda sweep."""
    networks = get_suite_networks(suite_name)
    all_results = {}

    for net_name, net_info in networks.items():
        print(f"\n  Testing {net_name}...", end="", flush=True)

        net_constructor = net_info["constructor"]
        try:
            net = net_constructor()
            network = build_network_from_pandapower(net, include_pv=True)
        except Exception as e:
            print(f" ERROR (skipping): {e}")
            continue

        tpf_results = test_tpf_lambda_sweep(
            network, lambdas, max_outer=max_outer, verbose=False
        )
        nr_results = test_nr_lambda_sweep(net, lambdas, verbose=False)

        tpf_conv = np.sum(tpf_results["converged_per_scenario"])
        nr_conv = np.sum(nr_results["converged_per_scenario"])
        print(f" TPF {tpf_conv}/{len(lambdas)}, NR {nr_conv}/{len(lambdas)}")

        all_results[net_name] = {
            "n_buses": network.n_buses,
            "n_pv": network.n_pv,
            "tpf_n_converged": int(tpf_conv),
            "nr_n_converged": int(nr_conv),
            "tpf_time_ms": tpf_results["elapsed_time_s"] * 1000,
            "nr_time_ms": nr_results["elapsed_time_s"] * 1000,
        }

    print_suite_summary(all_results, suite_name, lambdas)

    return all_results


def print_suite_summary(all_results: dict, suite_name: str, lambdas: np.ndarray):
    """Print summary table for suite results."""
    print(f"\n{'='*120}")
    print(f"  SUITE SUMMARY: {suite_name}")
    print(f"{'='*120}")

    header = (
        f"  {'Network':<25} {'Buses':<7} {'PV':<4} "
        f"{'TPF Conv':>10} {'NR Conv':>10} {'TPF Time':>12} {'NR Time':>12}"
    )
    print(header)
    print(f"  {'-'*118}")

    for net_name, data in all_results.items():
        print(
            f"  {net_name:<25} {data['n_buses']:<7} {data['n_pv']:<4} "
            f"{data['tpf_n_converged']:>10}/{len(lambdas)} "
            f"{data['nr_n_converged']:>10}/{len(lambdas)} "
            f"{data['tpf_time_ms']:>10.1f}ms "
            f"{data['nr_time_ms']:>10.1f}ms"
        )

    print(f"  {'-'*118}")

    avg_tpf_conv = np.mean([d["tpf_n_converged"] / len(lambdas) for d in all_results.values()])
    avg_nr_conv = np.mean([d["nr_n_converged"] / len(lambdas) for d in all_results.values()])
    avg_tpf_time = np.mean([d["tpf_time_ms"] for d in all_results.values()])
    avg_nr_time = np.mean([d["nr_time_ms"] for d in all_results.values()])

    print(
        f"  Avg Convergence: TPF {100*avg_tpf_conv:.1f}%, NR {100*avg_nr_conv:.1f}%  |  "
        f"Avg Time: TPF {avg_tpf_time:.1f}ms, NR {avg_nr_time:.1f}ms"
    )
    print(f"{'='*120}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test Load Factors lambda for Tensor Power Flow"
    )
    parser.add_argument(
        "--suite", type=str, default=None,
        help=f"Suite name. Valid: {VALID_SUITES}"
    )
    parser.add_argument(
        "--network", type=str, default=None,
        help="Single network name (overrides --suite)"
    )
    parser.add_argument(
        "--lambdas", type=str, default=None,
        help="Comma-separated lambda values (e.g., '0.3,0.5,0.7,1.0,1.5,2.0')"
    )
    parser.add_argument(
        "--lambda-min", type=float, default=0.3,
        help="Lambda min (default: 0.3)"
    )
    parser.add_argument(
        "--lambda-max", type=float, default=2.0,
        help="Lambda max (default: 2.0)"
    )
    parser.add_argument(
        "--lambda-steps", type=int, default=18,
        help="Number of lambda steps (default: 18)"
    )
    parser.add_argument(
        "--max-outer", type=int, default=30,
        help="Max outer iterations (default: 30)"
    )
    parser.add_argument(
        "--f3-study", action="store_true",
        help="Run F3 study (lambda × n_PV ratio grid search)"
    )
    parser.add_argument(
        "--f3-nodes", type=int, default=100,
        help="Base number of nodes for F3 study (default: 100)"
    )
    parser.add_argument(
        "--rx-study", action="store_true",
        help="Run R/X ratio study (lambda × R/X grid search)"
    )
    parser.add_argument(
        "--rx-nodes", type=int, default=100,
        help="Base number of nodes for R/X study (default: 100)"
    )
    parser.add_argument(
        "--show-plot", action="store_true",
        help="Show plots"
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="Save plot to file"
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.lambdas:
        lambdas = np.array([float(x) for x in args.lambdas.split(",")])
    else:
        lambdas = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)

    print(f"\n{'#'*80}")
    print(f"  LOAD FACTOR LAMBDA TEST")
    if args.network:
        print(f"  Network: {args.network}")
    elif args.f3_study:
        print(f"  Study: F3 (lambda x n_PV ratio)")
        print(f"  Base nodes: {args.f3_nodes}")
    elif args.rx_study:
        print(f"  Study: R/X ratio (lambda x R/X)")
        print(f"  Base nodes: {args.rx_nodes}")
    else:
        print(f"  Suite: {args.suite or 'quick'}")
    print(f"  Lambda range: [{lambdas[0]:.2f}, {lambdas[-1]:.2f}] with {len(lambdas)} steps")
    print(f"{'#'*80}")

    if args.f3_study:
        results = run_f3_study(
            base_nodes=args.f3_nodes,
            lambdas=lambdas,
            show_plot=args.show_plot,
            verbose=args.verbose,
        )
        if args.save_json:
            json_results = {
                "f3_study": {
                    "base_nodes": args.f3_nodes,
                    "lambdas": lambdas.tolist(),
                    "pv_ratios": results["pv_ratios"],
                    "results": results["results"].tolist(),
                    "converged": results["converged"].tolist(),
                }
            }
            with open(args.save_json, "w") as f:
                json.dump(json_results, f, indent=2, default=str)
            print(f"\n  Results saved to: {args.save_json}")
        return

    if args.rx_study:
        results = run_rx_ratio_study(
            base_nodes=args.rx_nodes,
            lambdas=lambdas,
            show_plot=args.show_plot,
            verbose=args.verbose,
        )
        if args.save_json:
            json_results = {
                "rx_study": {
                    "base_nodes": args.rx_nodes,
                    "lambdas": lambdas.tolist(),
                    "rx_ratios": results["rx_ratios"],
                    "results": results["results"].tolist(),
                    "converged": results["converged"].tolist(),
                }
            }
            with open(args.save_json, "w") as f:
                json.dump(json_results, f, indent=2, default=str)
            print(f"\n  Results saved to: {args.save_json}")
        return

    if args.network:
        results = test_single_network(
            args.network, lambdas,
            max_outer=args.max_outer,
            show_plot=args.show_plot,
            verbose=args.verbose,
        )
    else:
        suite = args.suite or "quick"
        results = test_suite(
            suite, lambdas,
            max_outer=args.max_outer,
            show_plot=args.show_plot,
            verbose=args.verbose,
        )

    if args.save_json:
        json_results = {"lambdas": lambdas.tolist()}
        if args.network:
            json_results["network_results"] = results
        else:
            json_results["suite_results"] = results
        with open(args.save_json, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()