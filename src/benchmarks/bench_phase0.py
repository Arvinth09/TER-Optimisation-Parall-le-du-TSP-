"""
Script de benchmarking pour la Phase 0 (Baseline).
Compare le Nearest Neighbor (Simple et Best) avec le 2-opt Séquentiel.
Utilisé pour générer les données du tableau comparatif dans le rapport.
"""

import time
import os
import sys

# Charger l'instance par défaut (Berlin52)
DEFAULT_INSTANCE = "data/berlin52.tsp"
OPTIMUM_BERLIN52 = 7542

from instance import TSPInstance
from solver import solve_nn, solve_2opt, calculate_tour_cost

def run_benchmark(instance_path=DEFAULT_INSTANCE, optimum=OPTIMUM_BERLIN52):
    if not os.path.exists(instance_path):
        print(f"Erreur : Le fichier {instance_path} est introuvable.")
        return

    print(f"--- Benchmark Phase 0 : {instance_path} ---")
    instance = TSPInstance(instance_path)
    
    # 1. Nearest Neighbor (Starting at node 0)
    start_time = time.perf_counter()
    tour_nn_0, dist_nn_0 = solve_nn(instance, start_node=0)
    time_nn_0 = (time.perf_counter() - start_time) * 1000
    gap_nn_0 = ((dist_nn_0 / optimum) - 1) * 100
    
    print(f"[1/3] NN (Start 0)  : Done. Distance = {dist_nn_0:.0f}, Gap = {gap_nn_0:.2f}%, Time = {time_nn_0:.2f} ms")

    # 2. Best Nearest Neighbor (Testing all possible starting nodes)
    start_time = time.perf_counter()
    best_dist_nn = float('inf')
    best_tour_nn = None
    for i in range(instance.dimension):
        tour, dist = solve_nn(instance, start_node=i)
        if dist < best_dist_nn:
            best_dist_nn = dist
            best_tour_nn = tour
    time_best_nn = (time.perf_counter() - start_time) * 1000
    gap_best_nn = ((best_dist_nn / optimum) - 1) * 100
    
    print(f"[2/3] Best NN (All) : Done. Distance = {best_dist_nn:.0f}, Gap = {gap_best_nn:.2f}%, Time = {time_best_nn:.2f} ms")

    # 3. 2-opt Séquentiel (Starting from the best NN tour)
    start_time = time.perf_counter()
    tour_2opt, dist_2opt = solve_2opt(instance, best_tour_nn)
    time_2opt = (time.perf_counter() - start_time) * 1000
    gap_2opt = ((dist_2opt / optimum) - 1) * 100
    
    print(f"[3/3] 2-opt Seq     : Done. Distance = {dist_2opt:.0f}, Gap = {gap_2opt:.2f}%, Time = {time_2opt:.2f} ms")
    print("-" * 40)
    
    # Résumé formaté
    print("\nDonnées pour le tableau LaTeX :")
    print(f"NN (0) & {dist_nn_0:.0f} & {gap_nn_0:.2f}\\% & {time_nn_0:.2f} ms \\\\")
    print(f"Best NN & {best_dist_nn:.0f} & {gap_best_nn:.2f}\\% & {time_best_nn:.2f} ms \\\\")
    print(f"2-opt Seq & {dist_2opt:.0f} & {gap_2opt:.2f}\\% & {time_2opt:.2f} ms \\\\")

if __name__ == "__main__":
    run_benchmark()
