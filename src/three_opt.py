"""
Phase 3 : Recherche locale 3-opt avec Parallélisation Multi-processus.

Ce module implémente l'algorithme 3-opt (First Improvement) restreint au voisinage K-NN.
Il supporte une exécution séquentielle et une exécution parallèle via multiprocessing.
"""

import os
import time
import argparse
from typing import Any, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Array

import numpy as np

from instance import TSPInstance
from solver import Tour, calculate_tour_cost, solve_nn, solve_2opt
from two_opt_knn import compute_knn, DELTA_EPS, solve_2opt_lot_knn


# --- INITIALISEUR DE POOL ---
_GLOBAL_INSTANCE = None
_GLOBAL_KNN = None
_GLOBAL_TOUR_SHARED = None

def init_worker(instance: TSPInstance, knn: list[list[int]], shared_tour):
    """
    Initialise chaque processus 'worker' du pool.
    Injecte les données lourdes (instance TSP, voisins K-NN, mémoire partagée) 
    une seule fois au démarrage pour éviter de les copier à chaque calcul.
    """
    global _GLOBAL_INSTANCE, _GLOBAL_KNN, _GLOBAL_TOUR_SHARED
    _GLOBAL_INSTANCE = instance
    _GLOBAL_KNN = knn
    _GLOBAL_TOUR_SHARED = shared_tour

# --- LOGIQUE 3-OPT ---

def appliquer_3opt_mouvement(tour: Tour, i: int, j: int, k: int, cas: int) -> Tour:
    """
    Réorganise le trajet en appliquant l'une des 4 reconnexions 3-opt "pures".
    Segments : A=[0..i-1], B=[i..j], C=[j+1..k], D=[k+1..n-1]
    Elle découpe le tour en 4 segments aux points i, j, k et les réassemble 
    selon le cas choisi (inversions ou déplacements de blocs).
    """
    A = tour[:i]
    B = tour[i : j + 1]
    C = tour[j + 1 : k + 1]
    D = tour[k + 1 :]

    if cas == 1: return A + B[::-1] + C + D
    if cas == 2: return A + B + C[::-1] + D
    if cas == 3: return A + B[::-1] + C[::-1] + D
    if cas == 4: return A + C + B + D
    if cas == 5: return A + C + B[::-1] + D
    if cas == 6: return A + C[::-1] + B + D
    if cas == 7: return A + C[::-1] + B[::-1] + D
    return tour

def appliquer_4opt_double_bridge(tour: Tour, i: int, j: int, k: int, l: int) -> Tour:
    """
    Applique un mouvement 4-opt de type 'Double-Bridge'.
    Découpe le tour en 4 segments aux index i, j, k, l et les réassemble 
    dans l'ordre A-D-C-B. Ce mouvement est crucial pour s'évader des minima locaux.
    Indices attendus : 0 < i < j < k < l < n
    """
    A = tour[:i]
    B = tour[i:j]
    C = tour[j:k]
    D = tour[k:l]
    E = tour[l:]
    
    # Reconnexion Double-Bridge : A-D-C-B-E
    return A + D + C + B + E

def calcul_gain_3opt(instance: TSPInstance, tour: Tour, i: int, j: int, k: int) -> tuple[float, int]:
    """
    Analyse les reconnexions 3-opt pour un triplet (i, j, k) donné.
    Elle compare le gain de distance pour chaque cas (1 à 7) et renvoie le
    meilleur gain trouvé ainsi que le numéro du cas correspondant.
    """
    gains = calcul_tous_gains_3opt(instance, tour, i, j, k)
    best_cas = max(gains, key=gains.get)
    return gains[best_cas], best_cas

def calcul_tous_gains_3opt(instance: TSPInstance, tour: Tour, i: int, j: int, k: int) -> dict[int, float]:
    """
    Calcul détaillé des gains pour chaque type de mouvement 3-opt (cas 1 à 7).
    Calcule la différence entre le coût des 3 arêtes supprimées et celui des
    3 nouvelles arêtes créées pour chaque scénario.
    """
    n = len(tour)
    s1_last, s2_first = tour[i - 1], tour[i]
    s2_last, s3_first = tour[j], tour[j + 1]
    s3_last, s4_first = tour[k], tour[(k + 1) % n]

    current_dist = (instance.get_distance(s1_last, s2_first) + 
                    instance.get_distance(s2_last, s3_first) + 
                    instance.get_distance(s3_last, s4_first))

    return {
        1: current_dist - (instance.get_distance(s1_last, s2_last) + instance.get_distance(s2_first, s3_first) + instance.get_distance(s3_last, s4_first)),
        2: current_dist - (instance.get_distance(s1_last, s2_first) + instance.get_distance(s2_last, s3_last) + instance.get_distance(s3_first, s4_first)),
        3: current_dist - (instance.get_distance(s1_last, s2_last) + instance.get_distance(s2_first, s3_last) + instance.get_distance(s3_first, s4_first)),
        4: current_dist - (instance.get_distance(s1_last, s3_first) + instance.get_distance(s3_last, s2_first) + instance.get_distance(s2_last, s4_first)),
        5: current_dist - (instance.get_distance(s1_last, s3_first) + instance.get_distance(s3_last, s2_last) + instance.get_distance(s2_first, s4_first)),
        6: current_dist - (instance.get_distance(s1_last, s3_last) + instance.get_distance(s3_first, s2_first) + instance.get_distance(s2_last, s4_first)),
        7: current_dist - (instance.get_distance(s1_last, s3_last) + instance.get_distance(s3_first, s2_last) + instance.get_distance(s2_first, s4_first))
    }


def solve_3opt_knn_sequential(instance: TSPInstance, initial_tour: Tour, k_neighbors: int, verbose: bool = True) -> tuple[Tour, float]:
    """
    Version classique (séquentielle) du solver 3-opt.
    Explore le voisinage en utilisant les K plus proches voisins pour aller plus vite.
    Elle applique la première amélioration (First Improvement) qu'elle rencontre.
    """
    tour = list(initial_tour)
    n = instance.dimension
    knn = compute_knn(instance, k_neighbors)
    best_cost = calculate_tour_cost(instance, tour)
    
    if verbose:
        print(f"Début 3-opt K-NN séquentiel (k={k_neighbors})...")

    # Mapping Ville -> Position dans le tour
    pos = [0] * n
    for p, city in enumerate(tour):
        pos[city] = p

    improvement = True
    while improvement:
        improvement = False
        
        # On parcourt les positions de coupure i
        for i in range(1, n - 4):
            city_i_prev = tour[i - 1]
            city_i = tour[i]
            
            for neighbor_c in knn[city_i_prev]:
                j = pos[neighbor_c]
                # Non-adjacence : j doit être après i+1
                if j < i + 2 or j >= n - 2: continue
                
                for neighbor_e in knn[city_i]:
                    k = pos[neighbor_e]
                    # Non-adjacence : k doit être après j+1
                    if k < j + 2 or k >= n - 1: continue
                    
                    gain, cas = calcul_gain_3opt(instance, tour, i, j, k)
                    if gain > DELTA_EPS:
                        tour = appliquer_3opt_mouvement(tour, i, j, k, cas)
                        # Mise à jour des positions
                        for p_idx in range(i, n): pos[tour[p_idx]] = p_idx
                        best_cost -= gain
                        improvement = True
                        break
                if improvement: break
            if improvement: break
            
    return tour, best_cost

# --- CORE PARALLELE ---

def search_3opt_chunk(chunk_start: int, chunk_end: int):
    """
    Fonction exécutée par un 'worker' pour explorer une tranche (chunk) du trajet.
    Elle cherche le mouvement le plus rentable dans sa zone en lisant le trajet 
    directement depuis la mémoire partagée pour gagner en rapidité.
    """
    global _GLOBAL_INSTANCE, _GLOBAL_KNN, _GLOBAL_TOUR_SHARED
    n = _GLOBAL_INSTANCE.dimension
    
    # Lecture du tour depuis la mémoire partagée
    tour_local = list(_GLOBAL_TOUR_SHARED)
    
    pos = [0] * n
    for p, city in enumerate(tour_local): pos[city] = p
    
    best_gain = 0.0
    best_move = None

    for i in range(chunk_start, chunk_end):
        if i >= n - 4: break
        
        city_i_prev = tour_local[i - 1]
        city_i = tour_local[i]
        
        for neighbor_c in _GLOBAL_KNN[city_i_prev]:
            j = pos[neighbor_c]
            if j < i + 2 or j >= n - 2: continue
            for neighbor_e in _GLOBAL_KNN[city_i]:
                k = pos[neighbor_e]
                if k < j + 2 or k >= n - 1: continue
                
                gain, cas = calcul_gain_3opt(_GLOBAL_INSTANCE, tour_local, i, j, k)
                if gain > best_gain + DELTA_EPS:
                    best_gain = gain
                    best_move = (i, j, k, cas)
                    
    return best_gain, best_move

def solve_3opt_knn_parallel(instance: TSPInstance, initial_tour: Tour, k_neighbors: int, num_processes: int = 4) -> tuple[Tour, float]:
    """
    Version haute performance (parallèle) du solver 3-opt.
    Découpe le travail en plusieurs morceaux et utilise tous les cœurs du CPU.
    Elle applique la meilleure amélioration (Best Improvement) d'un ensemble de tests.
    """
    tour = list(initial_tour)
    n = instance.dimension
    knn = compute_knn(instance, k_neighbors)
    best_cost = calculate_tour_cost(instance, tour)
    
    shared_tour = Array('i', tour)
    
    print(f"Lancement 3-opt Parallèle ({num_processes} processus, k={k_neighbors}, Shared Memory)...")

    with ProcessPoolExecutor(max_workers=num_processes, initializer=init_worker, initargs=(instance, knn, shared_tour)) as executor:
        while True:
            # On met à jour la mémoire partagée directement
            shared_tour[:] = tour
            
            # Découpage du domaine d'indices i en chunks
            chunk_size = max(1, (n - 2) // (num_processes * 4))
            chunks = [(i, min(i + chunk_size, n - 2)) for i in range(1, n - 2, chunk_size)]
            
            results = list(executor.map(search_3opt_chunk, [c[0] for c in chunks], [c[1] for c in chunks]))
            
            best_iteration_gain = 0.0
            best_iteration_move = None
            
            for gain, move in results:
                if gain > best_iteration_gain:
                    best_iteration_gain = gain
                    best_iteration_move = move
            
            if best_iteration_gain <= DELTA_EPS:
                break
                
            i, j, k, cas = best_iteration_move
            tour = appliquer_3opt_mouvement(tour, i, j, k, cas)
            best_cost -= best_iteration_gain
            
    return tour, best_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP 3-opt Local Search")
    parser.add_argument("--instance", type=str, default="data/berlin52.tsp", help="Fichier TSP")
    parser.add_argument("--k", type=int, default=15, help="Nombre de voisins K-NN")
    parser.add_argument("--parallel", action="store_true", help="Activer le parallélisme")
    parser.add_argument("--procs", type=int, default=4, help="Nombre de processus")
    parser.add_argument("--bench", action="store_true", help="Lancer le benchmark comparatif")
    
    args = parser.parse_args()
    
    inst_path = args.instance
    if not os.path.exists(inst_path):
        # Fallback si exécuté depuis src/
        inst_path = os.path.join("..", inst_path) if os.path.exists(os.path.join("..", inst_path)) else inst_path

    if not os.path.exists(inst_path):
        print(f"Erreur : Impossible de trouver l'instance {args.instance}")
        exit(1)

    instance = TSPInstance(inst_path)
    
    if args.bench:
        print(f"=== BENCHMARK PHASE 3 : {instance.name} ===")
        tour_nn, cost_nn = solve_nn(instance, start_node=0)
        
        # 1. 2-opt K-NN (Phase 2)
        t_2opt = time.perf_counter()
        tour_2o, cost_2o = solve_2opt_lot_knn(instance, tour_nn, args.k)
        d_2opt = time.perf_counter() - t_2opt
        print(f"2-opt lot + K-NN (k={args.k}) : {cost_2o:.2f} in {d_2opt:.4f}s")
        
        # 2. 3-opt Séquentiel (Phase 3)
        t_3seq = time.perf_counter()
        tour_3s, cost_3s = solve_3opt_knn_sequential(instance, tour_nn, args.k)
        d_3seq = time.perf_counter() - t_3seq
        print(f"3-opt K-NN Séquentiel : {cost_3s:.2f} in {d_3seq:.4f}s")
        
        # 3. 3-opt Parallèle (Phase 3)
        t_3par = time.perf_counter()
        tour_3p, cost_3p = solve_3opt_knn_parallel(instance, tour_nn, args.k, args.procs)
        d_3par = time.perf_counter() - t_3par
        print(f"3-opt K-NN Parallèle ({args.procs} procs) : {cost_3p:.2f} in {d_3par:.4f}s")
        
        speedup = d_3seq / d_3par if d_3par > 0 else 0
        gain_vs_2opt = cost_2o - cost_3p
        
        print("\n--- SYNTHÈSE ---")
        print(f"Speedup Multi-processus : {speedup:.2f}x")
        print(f"Gain Qualité 3-opt vs 2-opt : {gain_vs_2opt:.2f} unités")
        
    else:
        print(instance)
        tour_nn, cost_nn = solve_nn(instance, start_node=0)
        print(f"Coût Initial (NN) : {cost_nn:.2f}")
        
        t0 = time.perf_counter()
        if args.parallel:
            final_tour, final_cost = solve_3opt_knn_parallel(instance, tour_nn, args.k, args.procs)
        else:
            final_tour, final_cost = solve_3opt_knn_sequential(instance, tour_nn, args.k)
        duration = time.perf_counter() - t0
        
        print(f"\n--- RÉSULTATS ---")
        print(f"Coût Final : {final_cost:.2f}")
        print(f"Amélioration : {cost_nn - final_cost:.2f}")
        print(f"Temps : {duration:.2f} s")
