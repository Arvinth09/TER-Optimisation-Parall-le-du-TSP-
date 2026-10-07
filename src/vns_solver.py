"""
Phase 3.5 : Orchestrateur VNS (Variable Neighborhood Search).
Ce module combine le 2-opt, le 3-opt complet et le Double-Bridge (4-opt)
pour créer une heuristique capable d'échapper aux minima locaux.
"""

import time
import random
import argparse

from instance import TSPInstance
from solver import Tour, calculate_tour_cost
from two_opt_knn import solve_2opt_lot_knn, DELTA_EPS
from three_opt import solve_3opt_knn_parallel


def appliquer_4opt_double_bridge(tour: Tour, i: int, j: int, k: int, l: int) -> Tour:
    """
    Applique un mouvement 4-opt de type 'Double-Bridge'.
    Découpe le tour en 4 segments aux index i, j, k, l et les réassemble
    dans l'ordre A-D-C-B-E.
    Indices attendus : 0 < i < j < k < l < n
    """
    A = tour[:i]
    B = tour[i:j]
    C = tour[j:k]
    D = tour[k:l]
    E = tour[l:]

    return A + D + C + B + E

def perturbation_double_bridge(tour: Tour) -> Tour:
    """
    Applique une perturbation aléatoire de type Double-Bridge.
    Choisit 4 points de coupure aléatoires et réorganise les segments.
    """
    n = len(tour)
    if n < 8: return tour # Trop petit pour un double-bridge
    
    # On choisit 4 indices distincts et on les trie
    indices = sorted(random.sample(range(1, n - 1), 4))
    return appliquer_4opt_double_bridge(tour, *indices)

def solve_vns(instance: TSPInstance, initial_tour: Tour, k_neighbors: int, 
              max_kicks: int = 5, verbose: bool = True) -> tuple[Tour, float]:
    """
    Algorithme VNS :
    1. Descente locale via 2-opt puis 3-opt (parallèle).
    2. Si bloqué, on applique un 'Kick' (Double-Bridge) pour changer de zone.
    3. On recommence jusqu'à épuisement du nombre de kicks autorisés.
    """
    current_tour = list(initial_tour)
    best_tour = list(initial_tour)
    best_cost = calculate_tour_cost(instance, best_tour)
    
    if verbose:
        print(f"--- Lancement VNS ---")
        print(f"Coût initial : {best_cost:.2f}")

    kick_count = 0
    while kick_count <= max_kicks:
        if verbose and kick_count > 0:
            print(f"\n[Kick {kick_count}/{max_kicks}] Tentative d'évasion...")

        # --- PHASE DE DESCENTE (Local Search) ---
        
        # 1. Nettoyage rapide via 2-opt (lot compatible) + K-NN
        current_tour, cost = solve_2opt_lot_knn(instance, current_tour, k_neighbors)
        
        # 2. Affinement intense via 3-opt complet (Parallèle)
        current_tour, cost = solve_3opt_knn_parallel(instance, current_tour, k_neighbors, num_processes=4)
        
        # --- VERIFICATION AMELIORATION ---
        if cost < best_cost - DELTA_EPS:
            if verbose:
                print(f"  Amélioration trouvée ! {best_cost:.2f} -> {cost:.2f}")
            best_tour = list(current_tour)
            best_cost = cost
            # Si on a trouvé une amélioration, on peut soit reset les kicks, 
            # soit continuer. Ici on reset pour explorer à fond cette nouvelle zone.
            kick_count = 0 
        else:
            # On est dans un minimum local dont le 3-opt ne peut pas sortir.
            kick_count += 1
            if kick_count <= max_kicks:
                current_tour = perturbation_double_bridge(best_tour)

    if verbose:
        print(f"\n--- Fin VNS ---")
        print(f"Coût final : {best_cost:.2f}")
        
    return best_tour, best_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP VNS Solver")
    parser.add_argument("--instance", type=str, default="data/berlin52.tsp", help="Fichier TSP")
    parser.add_argument("--k", type=int, default=15, help="Voisins K-NN")
    parser.add_argument("--kicks", type=int, default=5, help="Nombre de perturbations")
    
    args = parser.parse_args()
    
    instance = TSPInstance(args.instance)
    from solver import solve_nn
    tour_nn, _ = solve_nn(instance, start_node=0)
    
    t0 = time.perf_counter()
    final_tour, final_cost = solve_vns(instance, tour_nn, args.k, args.kicks)
    duration = time.perf_counter() - t0
    
    print(f"Temps total : {duration:.2f}s")
