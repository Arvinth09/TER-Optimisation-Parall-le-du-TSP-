import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from instance import TSPInstance
from solver import solve_nn, solve_best_nn

def compare_nn():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'berlin52.tsp'))
    instance = TSPInstance(file_path)
    
    print(f"--- Comparaison NN sur {instance.name} ---")
    
    # NN classique (départ 0)
    _, cost0 = solve_nn(instance, start_node=0)
    print(f"NN (Départ ville 0) : {cost0:.2f}")
    
    # Best NN (tous départs)
    best_tour, best_cost = solve_best_nn(instance)
    print(f"Meilleur NN trouvé : {best_cost:.2f}")
    
    gain = cost0 - best_cost
    print(f"Gain réalisé : {gain:.2f} units")
    print(f"Nouvel écart à l'optimal (7542) : {((best_cost - 7542)/7542)*100:.2f}%")

if __name__ == "__main__":
    compare_nn()
