import time
import os
import sys

# Ajout du dossier src au path pour l'import
sys.path.append(os.path.join(os.getcwd(), 'src'))

from instance import TSPInstance
from solver import solve_2opt, solve_nn
from two_opt_knn import solve_2opt_lot_knn

def run_bench():
    print("--- Benchmark Phase 1 & 2 (Batch + K-NN) ---")
    
    # 1. Mesure pour le tableau (Berlin52)
    inst_b52 = TSPInstance("data/berlin52.tsp")
    optimum_b52 = 7542
    tour_init, _ = solve_nn(inst_b52, start_node=0)
    
    t0 = time.perf_counter()
    _, cost_b52 = solve_2opt_lot_knn(inst_b52, tour_init, k=20, verbose=False)
    time_b52 = (time.perf_counter() - t0) * 1000
    gap_b52 = ((cost_b52 / optimum_b52) - 1) * 100
    
    print(f"\n[Berlin52] Batch+K-NN (k=20) : Cost={cost_b52:.1f}, Gap={gap_b52:.2f}%, Time={time_b52:.2f} ms")

    # 2. Mesure du Speedup sur une instance plus grande (ch150)
    print("\n[ch150] Calcul du Speedup Algorithmique...")
    inst_ch150 = TSPInstance("data/ch150.tsp")
    tour_init_150, _ = solve_nn(inst_ch150, start_node=0)
    
    # Temps séquentiel
    t0 = time.perf_counter()
    solve_2opt(inst_ch150, tour_init_150)
    t_seq = time.perf_counter() - t0
    
    # Temps Batch + K-NN
    t0 = time.perf_counter()
    solve_2opt_lot_knn(inst_ch150, tour_init_150, k=20, verbose=False)
    t_batch = time.perf_counter() - t0
    
    speedup = t_seq / t_batch
    print(f"Temps Séquentiel : {t_seq:.2f} s")
    print(f"Temps Batch+K-NN : {t_batch:.2f} s")
    print(f"SPEEDUP : x{speedup:.1f}")

if __name__ == "__main__":
    run_bench()
