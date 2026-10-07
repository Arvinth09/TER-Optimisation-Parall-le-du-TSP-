import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from instance import TSPInstance
    from solver import solve_nn, solve_best_nn, solve_2opt
    from visualizer import plot_tour
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    from instance import TSPInstance
    from solver import solve_nn, solve_best_nn, solve_2opt
    from visualizer import plot_tour

def generate_viz():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_dir, 'data', 'berlin52.tsp')
    
    instance = TSPInstance(file_path)
    
    # 1. NN Classique (Départ 0)
    save_path_0 = os.path.join(base_dir, 'results', 'nn_tour_start0_berlin52.png')
    tour0, _ = solve_nn(instance, start_node=0)
    print(f"Génération graphique NN (Départ 0)...")
    plot_tour(instance, tour0, title=f"NN Classique (Départ 0) - {instance.name}", save_path=save_path_0)
    
    # 2. Best NN (Toutes villes)
    save_path_best = os.path.join(base_dir, 'results', 'best_nn_tour_berlin52.png')
    tour_best, _ = solve_best_nn(instance)
    print(f"Génération graphique Best NN...")
    plot_tour(instance, tour_best, title=f"Best NN (Toutes villes) - {instance.name}", save_path=save_path_best)

    # 3. 2-opt (Optimisé)
    save_path_2opt = os.path.join(base_dir, 'results', '2opt_tour_berlin52.png')
    # On repart du NN classique pour voir l'effet du 2-opt
    tour_nn, _ = solve_nn(instance, start_node=0)
    tour_2opt, _ = solve_2opt(instance, tour_nn)
    print(f"Génération graphique 2-opt...")
    plot_tour(instance, tour_2opt, title=f"2-opt Séquentiel - {instance.name}", save_path=save_path_2opt)

if __name__ == "__main__":
    generate_viz()


