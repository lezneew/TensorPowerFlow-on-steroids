"""Test sparse vs. dense TPF solvers for consistency - PQ and PV networks."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp
from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers import TPFDenseSolver, TPFDensePVMethodA
from tpf.solvers import TPFSparseConstantPower, TPFSparsePVMethodA


def create_pq_network():
    """Create simple PQ-only network for testing."""
    net = pp.create_empty_network()
    
    pp.create_buses(net, 5, vn_kv=11.0)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    
    for i in range(4):
        pp.create_line(net, i, i + 1, std_type="NAYY 4x50 SE", length_km=1.0)
    
    for i in range(1, 5):
        pp.create_load(net, i, p_mw=0.1, q_mvar=0.05)
    
    return net


def create_pv_network():
    """Create network with PV buses."""
    net = pp.create_empty_network()
    
    pp.create_buses(net, 6, vn_kv=11.0)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    
    for i in range(5):
        pp.create_line(net, i, i + 1, std_type="NAYY 4x50 SE", length_km=1.0)
    
    # PQ loads at buses 1, 2, 3, 4
    for i in range(1, 5):
        pp.create_load(net, i, p_mw=0.1, q_mvar=0.05)
    
    # PV bus at bus 5 (generator with fixed P and V)
    pp.create_gen(net, 5, p_mw=0.05, vm_pu=1.0)
    
    return net


def test_sparse_vs_dense_pq():
    """Test sparse vs. dense for PQ network."""
    net = create_pq_network()
    network = build_network_from_pandapower(net, include_pv=False)
    
    print("\n=== PQ Network Test ===")
    print(f"Network: {network.n_buses} buses")
    
    # Dense
    dense_solver = TPFDenseSolver(tol=1e-8, max_iter=100)
    dense_result = dense_solver.solve(network)
    
    # Sparse
    sparse_solver = TPFSparseConstantPower(tol=1e-8, max_iter=100)
    sparse_result = sparse_solver.solve(network)
    
    voltage_diff = np.max(np.abs(dense_result.voltages - sparse_result.voltages))
    power_diff = np.max(np.abs(dense_result.s_slack - sparse_result.s_slack))
    
    print(f"Voltage diff: {voltage_diff:.2e}")
    print(f"Power diff:   {power_diff:.2e}")
    print(f"Dense conv: {dense_result.converged} ({dense_result.iterations} iters)")
    print(f"Sparse conv: {sparse_result.converged} ({sparse_result.iterations} iters)")
    
    assert voltage_diff < 1e-6, f"Voltage mismatch: {voltage_diff}"
    assert power_diff < 1e-6, f"Power mismatch: {power_diff}"
    print("[PASS] PQ test passed!")


def test_sparse_vs_dense_pv():
    """Test sparse vs. dense for PV network."""
    net = create_pv_network()
    network = build_network_from_pandapower(net, include_pv=True)
    
    print("\n=== PV Network Test ===")
    print(f"Network: {network.n_buses} buses, {network.n_pv} PV buses")
    
    # Dense
    dense_solver = TPFDensePVMethodA(tol=1e-8, max_iter_inner=50, max_iter_outer=20)
    dense_result = dense_solver.solve(network)
    
    # Sparse
    sparse_solver = TPFSparsePVMethodA(tol=1e-8, max_iter_inner=50, max_iter_outer=20)
    sparse_result = sparse_solver.solve(network)
    
    voltage_diff = np.max(np.abs(dense_result.voltages - sparse_result.voltages))
    power_diff = np.max(np.abs(dense_result.s_slack - sparse_result.s_slack))
    
    print(f"Voltage diff: {voltage_diff:.2e}")
    print(f"Power diff:   {power_diff:.2e}")
    print(f"Dense conv: {dense_result.converged} ({dense_result.iterations} iters)")
    print(f"Sparse conv: {sparse_result.converged} ({sparse_result.iterations} iters)")
    
    assert voltage_diff < 1e-6, f"Voltage mismatch: {voltage_diff}"
    assert power_diff < 1e-6, f"Power mismatch: {power_diff}"
    print("[PASS] PV test passed!")


if __name__ == "__main__":
    test_sparse_vs_dense_pq()
    test_sparse_vs_dense_pv()
    print("\n=== All tests passed! ===")