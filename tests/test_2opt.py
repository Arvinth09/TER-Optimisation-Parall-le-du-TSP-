import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from instance import TSPInstance
    from solver import solve_nn, solve_2opt
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    from instance import TSPInstance
    from solver import solve_nn, solve_2opt

def run_2opt_test():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_dir, 'data', 'berlin52.tsp')
    
    instance = TSPInstance(file_path)
    
    # 1. Génération du tour initial (NN ville 0)
    print("--- Phase 1 : Génération du tour initial (NN) ---")
    initial_tour, initial_cost = solve_nn(instance, start_node=0)
    print(f"Coût initial (NN) : {initial_cost:.2f}")
    
    # 2. Amélioration par 2-opt
    print("\n--- Phase 2 : Optimisation 2-opt ---")
    start_time = time.time()
    optimized_tour, optimized_cost = solve_2opt(instance, initial_tour)
    end_time = time.time()
    
    duration = end_time - start_time
    
    print("\n--- Résultats ---")
    print(f"Coût final (2-opt) : {optimized_cost:.2f}")
    print(f"Amélioration : {initial_cost - optimized_cost:.2f} units")
    print(f"Gain en pourcentage : {((initial_cost - optimized_cost)/initial_cost)*100:.2f}%")
    print(f"Temps de calcul : {duration:.4f} secondes")
    
    # Écart à l'optimal (7542)
    optimal = 7542
    gap = ((optimized_cost - optimal) / optimal) * 100
    print(f"Écart à l'optimal (7542) : {gap:.2f}%")

if __name__ == "__main__":
    run_2opt_test()
