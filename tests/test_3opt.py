import sys
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Ajout du dossier src au path pour permettre les imports directs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from instance import TSPInstance
from three_opt import (
    appliquer_3opt_mouvement, 
    calcul_gain_3opt, 
    calcul_tous_gains_3opt,
    solve_3opt_knn_sequential,
    solve_3opt_knn_parallel
)
from solver import calculate_tour_cost, Tour, solve_nn

class MockInstance:
    """Une instance simplifiée pour les tests unitaires."""
    def __init__(self):
        self.dimension = 10
    def get_distance(self, i, j):
        return float(abs(i - j))

def test_3opt_mouvements_structure():
    """Vérifie que les reconnexions 3-opt produisent toujours un tour valide."""
    tour = list(range(10))
    i, j, k = 2, 5, 8
    
    for cas in range(1, 8):
        nouveau_tour = appliquer_3opt_mouvement(tour, i, j, k, cas)
        assert len(nouveau_tour) == 10
        assert len(set(nouveau_tour)) == 10
        assert sorted(nouveau_tour) == list(range(10))

def test_3opt_gain_precision_individuelle():
    """Vérifie explicitement la formule de gain de chaque cas 4/5/6/7."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "berlin52.tsp")
    
    if not os.path.exists(file_path):
        pytest.skip("Instance berlin52.tsp non trouvée")
        
    instance = TSPInstance(file_path)
    tour = list(range(instance.dimension))
    # Indices respectant la non-adjacence
    i, j, k = 5, 15, 30
    
    gains_theoriques = calcul_tous_gains_3opt(instance, tour, i, j, k)
    
    for cas in range(1, 8):
        nouveau_tour = appliquer_3opt_mouvement(tour, i, j, k, cas)
        cout_initial = calculate_tour_cost(instance, tour)
        cout_final = calculate_tour_cost(instance, nouveau_tour)
        gain_reel = cout_initial - cout_final
        
        assert gains_theoriques[cas] == pytest.approx(gain_reel, rel=1e-5), f"Échec gain Cas {cas}"

def test_3opt_coherence_seq_par():
    """Vérifie que Seq et Par produisent des résultats cohérents sur le même tour initial."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "berlin52.tsp")
    
    if not os.path.exists(file_path):
        pytest.skip("Instance berlin52.tsp non trouvée")
        
    instance = TSPInstance(file_path)
    tour_nn, initial_cost = solve_nn(instance, start_node=0)
    
    # On limite k et les itérations pour un test rapide
    k = 10
    
    tour_seq, cost_seq = solve_3opt_knn_sequential(instance, tour_nn, k, verbose=False)
    tour_par, cost_par = solve_3opt_knn_parallel(instance, tour_nn, k, num_processes=2)
    
    # Les deux doivent au moins ne pas dégrader le tour initial
    assert cost_seq <= initial_cost + 1e-9
    assert cost_par <= initial_cost + 1e-9
    
    # La version Parallèle étant "Best Improvement", elle devrait souvent être égale ou meilleure que Seq (First Improvement)
    # ou au moins dans le même ordre de grandeur.
    assert cost_par <= cost_seq + 1000 # Marge large car les stratégies diffèrent (First vs Best)

def test_3opt_non_adjacence_loops():
    """Vérifie que les contraintes i < j+2 < k+2 sont bien pensées pour éviter les cas dégénérés."""
    # Ce test est plus une vérification logique du développeur
    n = 20
    count = 0
    for i in range(1, n - 4):
        for j in range(i + 2, n - 2):
            for k in range(j + 2, n - 1):
                count += 1
                assert j >= i + 2
                assert k >= j + 2
                assert (k + 1) % n != i - 1
    assert count > 0

def test_3opt_exclusion_fonctionnelle_spy():
    """Vérifie par espionnage (Mock) que le solver n'appelle JAMAIS de cas adjacents."""
    instance = MockInstance()
    # Tour simple : 0-1-2-3-4-5-6-7-8-9
    tour = list(range(10))
    k = 3
    
    # On patche calcul_gain_3opt pour enregistrer les appels 
    # ET on patche compute_knn pour éviter d'accéder aux coords du Mock
    with patch('three_opt.calcul_gain_3opt') as mock_gain, \
         patch('three_opt.compute_knn') as mock_knn:
        
        # Simulation d'un voisinage éloigné pour éviter les 'continue' de non-adjacence
        # i=1, j=(1+4)=5 (OK: 5 >= 1+2), k=(1+8)=8 (OK: 8 >= 5+2)
        mock_knn.return_value = [[(i+4)%10, (i+6)%10, (i+8)%10] for i in range(10)]
        # Simulation d'un gain nul pour parcourir tout le voisinage sans changer le tour
        mock_gain.return_value = (0.0, 4)
        
        # On lance une exécution séquentielle
        solve_3opt_knn_sequential(instance, tour, k, verbose=False)
        
        # Vérification des appels enregistrés
        assert mock_gain.called
        for call in mock_gain.call_args_list:
            # call[0] contient les arguments (instance, tour, i, j, k)
            i, j, k_idx = call[0][2], call[0][3], call[0][4]
            
            # Vérification stricte des contraintes de non-adjacence
            assert j >= i + 2, f"Adjacence détectée j={j} pour i={i}"
            assert k_idx >= j + 2, f"Adjacence détectée k={k_idx} pour j={j}"
            # Circularité (i-1 et k+1)
            n = 10
            assert (k_idx + 1) % n != i - 1, f"Adjacence circulaire détectée k={k_idx} pour i={i}"

def test_4opt_double_bridge_structure():
    """Vérifie que le mouvement Double-Bridge produit un tour valide."""
    from three_opt import appliquer_4opt_double_bridge
    tour = list(range(20))
    # Coupures : 2, 6, 10, 15
    nouveau_tour = appliquer_4opt_double_bridge(tour, 2, 6, 10, 15)
    
    assert len(nouveau_tour) == 20
    assert len(set(nouveau_tour)) == 20
    assert sorted(nouveau_tour) == list(range(20))
    # Vérifie que le tour est différent de l'original
    assert nouveau_tour != tour
