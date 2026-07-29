"""Performance comparison: sparse vs dense TPF solvers."""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower.networks as nw
from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers import TPFDenseSolver, TPFSparseConstantPower


def test_network(net_name: str, include_pv: bool = False):
    """Test a single network."""
    print(f"\n{'='*60}")
    print(f"Network: {net_name}")
    print(f"{'='*60}")
    
    # Build network
    net = getattr(nw, net_name)()
    network = build_network_from_pandapower(net, include_pv=include_pv)
    
    print(f"Buses: {network.n_buses}")
    print(f"Y_dd sparsity: {100 * (1 - network.Y_dd.size / (network.n_buses ** 2)):.1f}%")
    
    # Dense solver
    t0 = time.perf_counter()
    dense_solver = TPFDenseSolver(tol=1e-8, max_iter=100)
    dense_result = dense_solver.solve(network)
    dense_time = time.perf_counter() - t0
    
    # Sparse solver
    t0 = time.perf_counter()
    sparse_solver = TPFSparseConstantPower(tol=1e-8, max_iter=100)
    sparse_result = sparse_solver.solve(network)
    sparse_time = time.perf_counter() - t0
    
    # Compare
    voltage_diff = np.max(np.abs(dense_result.voltages - sparse_result.voltages))
    
    print(f"\nDense:   {dense_time*1000:.2f} ms, {dense_result.iterations} iters")
    print(f"Sparse:  {sparse_time*1000:.2f} ms, {sparse_result.iterations} iters")
    print(f"Ratio:   {dense_time/sparse_time:.2f}x speedup")
    print(f"Voltage diff: {voltage_diff:.2e}")
    
    return network.n_buses, dense_time, sparse_time


def main():
    """Run benchmarks on various networks."""
    networks = [
        ("simple_four_buses", False),
        ("case9", False),
        ("case14", False),
        ("case30", False),
        ("case57", False),
        ("case118", False),
    ]
    
    results = []
    for net_name, include_pv in networks:
        try:
            n_buses, dense_time, sparse_time = test_network(net_name, include_pv)
            results.append((net_name, n_buses, dense_time, sparse_time))
        except Exception as e:
            print(f"[ERROR] {net_name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Network':<20} {'Buses':>8} {'Dense (ms)':>12} {'Sparse (ms)':>12} {'Speedup':>10}")
    print("-" * 60)
    for net_name, n_buses, dense_time, sparse_time in results:
        ratio = dense_time / sparse_time if sparse_time > 0 else 0
        print(f"{net_name:<20} {n_buses:>8} {dense_time*1000:>12.2f} {sparse_time*1000:>12.2f} {ratio:>10.2f}x")


if __name__ == "__main__":
    main()