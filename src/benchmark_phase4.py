"""
Phase 4 (Version Corrigée) : Script de Benchmark Scientifique.
Conforme au protocole : 5 runs, processus {1,2,4,8}, validation de tour et synthèse CSV.
"""

import os
import time
import csv
import random
import numpy as np
from typing import List, Dict

from instance import TSPInstance
from solver import solve_nn, calculate_tour_cost
from two_opt_knn import solve_2opt_lot_knn
from three_opt import solve_3opt_knn_parallel, solve_3opt_knn_sequential
from vns_solver import solve_vns

# --- CONFIGURATION MISE À JOUR ---
INSTANCES = [
    {"name": "berlin52", "file": "data/berlin52.tsp", "optimum": 7542},
    {"name": "ch150", "file": "data/ch150.tsp", "optimum": 6528},
    {"name": "pcb442", "file": "data/pcb442.tsp", "optimum": 50778},
    {"name": "pr1002", "file": "data/pr1002.tsp", "optimum": 259045},
]

ALGO_CONFIGS = [
    {"name": "NN", "func": "solve_nn", "procs": 1},
    {"name": "2-opt lot + K-NN", "func": "solve_2opt_lot_knn", "procs": 1},
    {"name": "3-opt Par (1 p)", "func": "solve_3opt_knn_parallel", "procs": 1},
    {"name": "3-opt Par (2 p)", "func": "solve_3opt_knn_parallel", "procs": 2},
    {"name": "3-opt Par (4 p)", "func": "solve_3opt_knn_parallel", "procs": 4},
    {"name": "3-opt Par (8 p)", "func": "solve_3opt_knn_parallel", "procs": 8},
    {"name": "VNS (Hybrid)", "func": "solve_vns", "kicks": 5, "procs": 4} # VNS utilise 4 procs par défaut
]

K_NEIGHBORS = 15
NB_RUNS = 5
SEED = 42

def validate_tour(instance: TSPInstance, tour: List[int]):
    """Vérifie la validité structurelle du tour."""
    if len(tour) != instance.dimension:
        raise ValueError(f"Taille incorrecte : {len(tour)} au lieu de {instance.dimension}")
    if len(set(tour)) != instance.dimension:
        raise ValueError("Doublons ou villes manquantes dans le tour")

def run_benchmark():
    os.makedirs("results", exist_ok=True)
    raw_path = "results/phase4_raw.csv"
    summary_path = "results/phase4_summary.csv"
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    raw_results = []
    
    with open(raw_path, mode='w', newline='') as csvfile:
        fieldnames = ['instance', 'algo', 'run', 'cost', 'time_s', 'gap_percent', 'procs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for inst_info in INSTANCES:
            if not os.path.exists(inst_info["file"]): continue
            
            print(f"\n>>> Benchmark Instance: {inst_info['name']} (Optimum: {inst_info['optimum']})")
            instance = TSPInstance(inst_info["file"])
            tour_initial, cost_nn = solve_nn(instance, start_node=0)
            
            for algo in ALGO_CONFIGS:
                print(f"  Algo: {algo['name']}...", end=" ", flush=True)
                for run in range(1, NB_RUNS + 1):
                    t_start = time.perf_counter()
                    
                    # Execution
                    f_name = algo["func"]
                    tour_obtenu = []
                    cost = 0.0
                    
                    if f_name == "solve_nn":
                        tour_obtenu, cost = solve_nn(instance, start_node=0)
                    elif f_name == "solve_2opt_lot_knn":
                        tour_obtenu, cost = solve_2opt_lot_knn(instance, tour_initial, K_NEIGHBORS)
                    elif f_name == "solve_3opt_knn_parallel":
                        tour_obtenu, cost = solve_3opt_knn_parallel(instance, tour_initial, K_NEIGHBORS, num_processes=algo["procs"])
                    elif f_name == "solve_vns":
                        tour_obtenu, cost = solve_vns(instance, tour_initial, K_NEIGHBORS, max_kicks=algo["kicks"], verbose=False)
                    
                    duration = time.perf_counter() - t_start
                    
                    # Validation
                    validate_tour(instance, tour_obtenu)
                    
                    gap = ((cost - inst_info["optimum"]) / inst_info["optimum"]) * 100
                    
                    row = {
                        'instance': inst_info['name'],
                        'algo': algo['name'],
                        'run': run,
                        'cost': cost,
                        'time_s': duration,
                        'gap_percent': gap,
                        'procs': algo["procs"]
                    }
                    writer.writerow(row)
                    raw_results.append(row)
                print(" OK")
                csvfile.flush()

    # --- GENERATION DU SUMMARY ---
    print("\n>>> Génération du fichier de synthèse (summary)...")
    summary_data = {}
    for r in raw_results:
        key = (r['instance'], r['algo'])
        if key not in summary_data:
            summary_data[key] = {'costs': [], 'times': [], 'gaps': [], 'procs': r['procs']}
        summary_data[key]['costs'].append(r['cost'])
        summary_data[key]['times'].append(r['time_s'])
        summary_data[key]['gaps'].append(r['gap_percent'])

    with open(summary_path, mode='w', newline='') as csvfile:
        fieldnames = ['instance', 'algo', 'procs', 'cost_mean', 'cost_min', 'cost_max', 'time_mean', 'gap_mean', 'std_dev_gap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (inst, algo_name), data in summary_data.items():
            writer.writerow({
                'instance': inst,
                'algo': algo_name,
                'procs': data['procs'],
                'cost_mean': np.mean(data['costs']),
                'cost_min': np.min(data['costs']),
                'cost_max': np.max(data['costs']),
                'time_mean': np.mean(data['times']),
                'gap_mean': np.mean(data['gaps']),
                'std_dev_gap': np.std(data['gaps'])
            })

    print(f"Terminé ! Synthèse disponible dans {summary_path}")

if __name__ == "__main__":
    run_benchmark()
