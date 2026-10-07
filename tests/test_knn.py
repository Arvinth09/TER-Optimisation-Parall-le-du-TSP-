import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from two_opt_knn import benchmark_knn, compute_knn, solve_2opt_lot_knn
    from instance import TSPInstance
    from solver import calculate_tour_cost, solve_nn
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
    from two_opt_knn import benchmark_knn, compute_knn, solve_2opt_lot_knn
    from instance import TSPInstance
    from solver import calculate_tour_cost, solve_nn


def run_knn_test() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(base_dir, "data", "berlin52.tsp")

    instance = TSPInstance(file_path)
    initial_tour, initial_cost = solve_nn(instance, start_node=0)

    print("--- Test du 2-opt lot + k-NN ---")
    print(f"Coût initial (NN) : {initial_cost:.2f}")

    knn = compute_knn(instance, k=10)
    if len(knn) != instance.dimension:
        print("Erreur : une liste de voisins par ville attendue.")
        return
    for ville, voisins in enumerate(knn):
        if ville in voisins:
            print(f"Erreur : la ville {ville} se trouve dans ses propres voisins.")
            return
        if len(voisins) != 10:
            print(f"Erreur : mauvaise taille de voisinage pour {ville}.")
            return

    final_tour, final_cost = solve_2opt_lot_knn(
        instance, initial_tour, k=10, knn=knn, verbose=False
    )
    print(f"Coût 2-opt lot + k-NN (k=10) : {final_cost:.2f}")

    if len(final_tour) != instance.dimension or len(set(final_tour)) != instance.dimension:
        print("Erreur : le tour final n'est pas valide.")
        return

    recompute = calculate_tour_cost(instance, final_tour)
    if abs(recompute - final_cost) > 1e-6:
        print("Erreur : coût incohérent avec le tour retourné.")
        return

    if final_cost >= initial_cost:
        print("Erreur : pas d'amélioration sur le tour NN.")
        return

    print("--- Mini-benchmark k dans {5, 10, 20} ---")
    benchmark_knn(instance, initial_tour, ks=[5, 10, 20], verbose=True)

    print("Succès : voisinages k-NN, amélioration et tour valides.")


if __name__ == "__main__":
    run_knn_test()
