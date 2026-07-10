import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tpf.generators.network_generator_salazar import SALAZAR_TEST_NETWORKS, SALAZAR_SCALING_NETWORKS
from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA

# Test a few Salazar networks with different PV counts
test_names = [
    "salazar_100bus_5pv",
    "salazar_100bus_10pv", 
    "salazar_200bus_10pv",
    "salazar_500bus_25pv",
    "salazar_1000bus_50pv",
]

for name in SALAZAR_SCALING_NETWORKS:
    # if name not in SALAZAR_TEST_NETWORKS:
    #     print(f"Skipping {name} - not in SALAZAR_TEST_NETWORKS")
    #     continue
    
    print(f"\n=== {name} ===")
    constructor = SALAZAR_SCALING_NETWORKS[name]["constructor"]
    net = constructor()
    
    # Build network
    network = build_network_from_pandapower(net, include_pv=True)
    
    if not network.has_pv:
        print(f"  No PV buses, skipping")
        continue
    
    print(f"  n_buses={network.n_buses}, n_pv={network.n_pv}")
    
    # Create solver
    solver = TPFDensePVMethodA(max_iter_outer=100, max_iter_inner=50, omega=0.5)
    
    # Simple test: constant power profile
    s_test = np.zeros((network.n_bus_phases, 1), dtype=np.complex128)
    
    # Run solver - this should print the diagnostic
    try:
        result = solver.solve_timeseries(network, s_test)
        print(f"  Converged: {result.converged}")
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone!")