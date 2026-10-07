import random

from instance import TSPInstance

Tour = list[int]


def solve_nn(
    instance: TSPInstance, start_node: int | None = None
) -> tuple[Tour, float]:
    """
    Génère un tour TSP en utilisant l'heuristique du plus proche voisin.
    """
    num_cities = instance.dimension

    if start_node is None:
        current_node = random.randint(0, num_cities - 1)
    else:
        current_node = start_node

    tour: Tour = [current_node]
    unvisited = set(range(num_cities))
    unvisited.remove(current_node)

    total_cost = 0.0

    while unvisited:
        next_node: int | None = None
        min_dist = float("inf")

        for candidate in unvisited:
            d = instance.get_distance(current_node, candidate)
            if d < min_dist:
                min_dist = d
                next_node = candidate

        assert next_node is not None
        tour.append(next_node)
        unvisited.remove(next_node)
        total_cost += min_dist
        current_node = next_node

    total_cost += instance.get_distance(tour[-1], tour[0])

    return tour, total_cost


def calculate_tour_cost(instance: TSPInstance, tour: Tour) -> float:
    """Calcul du coût total d'un tour donné pour vérification."""
    cost = 0.0
    for i in range(len(tour)):
        cost += instance.get_distance(tour[i], tour[(i + 1) % len(tour)])
    return cost


##ajout des fonctions two_opt_swap et solve_2opt  pour la partie 0.4
##

def two_opt_swap(tour: Tour, i: int, j: int) -> Tour:
    """
    Effectue un 2-opt swap : inverse le segment du tour entre les index i et j (inclus).
    """
    return tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]


def solve_2opt(instance: TSPInstance, initial_tour: Tour) -> tuple[Tour, float]:
    """
    Améliore un tour initial en utilisant l'algorithme 2-opt (First Improvement).
    """
    tour = list(initial_tour)
    best_cost = calculate_tour_cost(instance, tour)
    num_cities = instance.dimension

    improvement = True
    print(f"Début de l'optimisation 2-opt... Coût initial : {best_cost:.2f}")

    while improvement:
        improvement = False

        for i in range(1, num_cities - 1):
            for j in range(i + 1, num_cities):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % num_cities]

                current_dist = instance.get_distance(a, b) + instance.get_distance(c, d)
                new_dist = instance.get_distance(a, c) + instance.get_distance(b, d)

                if new_dist < current_dist - 0.00001:
                    tour = two_opt_swap(tour, i, j)
                    best_cost = calculate_tour_cost(instance, tour)
                    improvement = True
                    break
            if improvement:
                break

    print(f"Fin de l'optimisation 2-opt. Coût final : {best_cost:.2f}")
    return tour, best_cost


def solve_best_nn(instance: TSPInstance) -> tuple[Tour, float]:
    """
    Teste tous les points de départ possibles et retourne le meilleur tour trouvé.
    """
    best_tour: Tour | None = None
    best_cost = float("inf")

    print(f"Évaluation de {instance.dimension} points de départ...")

    for start_node in range(instance.dimension):
        tour, cost = solve_nn(instance, start_node=start_node)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    assert best_tour is not None
    return best_tour, best_cost
