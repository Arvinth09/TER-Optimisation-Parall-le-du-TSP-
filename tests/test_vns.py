import sys
import os
import pytest
sys.path.append(os.path.abspath("src"))

from instance import TSPInstance
from solver import solve_nn
from vns_solver import solve_vns

def test_vns_run_basic():
    """Vérifie que le solver VNS s'exécute sans erreur sur une petite instance."""
    if not os.path.exists("data/berlin52.tsp"):
        pytest.skip("Instance berlin52.tsp non trouvée")
        
    instance = TSPInstance("data/berlin52.tsp")
    tour_nn, cost_nn = solve_nn(instance, start_node=0)
    
    # On lance un VNS très court (1 kick)
    final_tour, final_cost = solve_vns(instance, tour_nn, k_neighbors=10, max_kicks=1, verbose=False)
    
    # Vérifications de base
    assert len(final_tour) == instance.dimension
    assert len(set(final_tour)) == instance.dimension
    # Le VNS doit être au moins aussi bon que le NN
    assert final_cost <= cost_nn
