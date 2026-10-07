import sys
import os

# Ajout du dossier src au chemin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from instance import TSPInstance
    from solver import solve_nn
except ImportError:
    # Cas où on est à la racine
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    from instance import TSPInstance
    from solver import solve_nn

def run_nn_test():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_dir, 'data', 'berlin52.tsp')
    
    if not os.path.exists(file_path):
        print(f"Erreur : le fichier {file_path} est introuvable.")
        return

    instance = TSPInstance(file_path)
    
    print(f"--- Résolution de {instance.name} via Nearest Neighbor ---")
    
    # On teste en partant de la ville 0
    tour, cost = solve_nn(instance, start_node=0)
    
    print(f"Tour généré (premières villes) : {tour[:10]}...")
    print(f"Nombre de villes dans le tour : {len(tour)}")
    print(f"Coût total calculé : {cost:.2f}")
    
    # Vérification : Berlin52 optimal est à 7542. 
    # Le NN donne généralement autour de ~8901.69 sur Berlin52.
    gap = ((cost - 7542) / 7542) * 100
    print(f"Ecart par rapport à l'optimal théorique (7542) : {gap:.2f}%")

if __name__ == "__main__":
    run_nn_test()
