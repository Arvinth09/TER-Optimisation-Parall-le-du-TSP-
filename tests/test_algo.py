import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from two_opt_batch import (
        appliquer_lot_simultanement,
        construire_graphe_conflits,
        generer_candidats,
        mouvements_en_conflit,
        selectionner_lot_glouton,
        solve_2opt_lot,
    )
    from instance import TSPInstance
    from solver import calculate_tour_cost, solve_nn
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
    from two_opt_batch import (
        appliquer_lot_simultanement,
        construire_graphe_conflits,
        generer_candidats,
        mouvements_en_conflit,
        selectionner_lot_glouton,
        solve_2opt_lot,
    )
    from instance import TSPInstance
    from solver import calculate_tour_cost, solve_nn


def run_algo_test():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(base_dir, "data", "berlin52.tsp")

    instance = TSPInstance(file_path)
    initial_tour, initial_cost = solve_nn(instance, start_node=0)

    print("--- Test du lot compatible 2-opt ---")
    print(f"Coût initial (NN) : {initial_cost:.2f}")

    candidats = generer_candidats(instance, initial_tour)
    print(f"Candidats positifs : {len(candidats)}")

    graphe = construire_graphe_conflits(candidats)
    lot = selectionner_lot_glouton(candidats, graphe)
    print(f"Taille du lot compatible : {len(lot)}")

    for index_a in range(len(lot)):
        for index_b in range(index_a + 1, len(lot)):
            if mouvements_en_conflit(lot[index_a], lot[index_b]):
                print("Erreur : deux mouvements du lot sont incompatibles.")
                return

    batch_tour = appliquer_lot_simultanement(initial_tour, lot)
    batch_cost = calculate_tour_cost(instance, batch_tour)
    print(f"Coût après une application de lot : {batch_cost:.2f}")

    if batch_cost >= initial_cost:
        print("Erreur : le lot appliqué n'améliore pas le tour.")
        return

    final_tour, final_cost = solve_2opt_lot(instance, initial_tour, verbose=False)
    print(f"Coût final après itérations : {final_cost:.2f}")

    if len(final_tour) != instance.dimension or len(set(final_tour)) != instance.dimension:
        print("Erreur : le tour final n'est pas valide.")
        return

    if final_cost >= initial_cost:
        print("Erreur : l'algorithme par lots n'améliore pas le tour initial.")
        return

    print("Succès : lot compatible, application valide, amélioration confirmée.")


if __name__ == "__main__":
    run_algo_test()
