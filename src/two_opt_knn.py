"""
Travail (2) du TER : réduction du voisinage 2-opt aux k plus proches voisins.

Idée :
- Pour chaque ville, on précalcule la liste de ses k plus proches voisins
  (candidate list géométrique, cf. énoncé "k plus proches voisins ou candidats
  par heuristiques").
- On ne génère les candidats 2-opt que parmi les paires autorisées par ces listes
  (au lieu d'énumérer tous les couples (i, j)).
- On réutilise la logique du Travail (1) : graphe de conflits, MIS glouton,
  application simultanée d'un lot compatible.
- On étudie le compromis k vs temps vs qualité sur berlin52.

Module : two_opt_knn.
"""

import os
import time

import numpy as np

from instance import TSPInstance
from solver import Tour, calculate_tour_cost, solve_2opt, solve_nn
from two_opt_batch import (
    TwoOptCandidat,
    appliquer_lot_simultanement,
    calcul_gain_2opt,
    construire_graphe_conflits,
    selectionner_lot_glouton,
    solve_2opt_lot,
)
from visualizer import plot_conflict_graph, plot_knn_tradeoff, plot_tour

DELTA_EPS = 1e-5


def compute_knn(instance: TSPInstance, k: int) -> list[list[int]]:
    """
    Pour chaque ville, retourne la liste de ses `k` plus proches voisins
    (indices 0-indexés), triés par distance croissante.

    Implémentation vectorisée :
    - calcule les distances au carré entre toutes les paires (i, j)
    - exclut i==j via +inf sur la diagonale
    - récupère les k plus proches via argpartition (sans trier toute la ligne)
      puis trie uniquement ces k voisins pour obtenir un ordre croissant.
    """
    coords = instance.coords
    assert coords is not None

    n = instance.dimension
    k_eff = min(k, n - 1)

    # (n, n, d) -> (n, n) distances au carré
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = (diff**2).sum(axis=-1)
    np.fill_diagonal(dist2, np.inf)

    # Indices des k plus petits éléments par ligne (ordre non garanti)
    nearest = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]

    # Tri des k voisins par distance croissante
    row_ids = np.arange(n)[:, None]
    order = np.argsort(dist2[row_ids, nearest], axis=1)
    nearest_sorted = nearest[row_ids, order]

    return [[int(x) for x in row] for row in nearest_sorted]


def generer_candidats_knn(
    instance: TSPInstance,
    tour: Tour,
    knn: list[list[int]],
) -> list[TwoOptCandidat]:
    """
    Génère les candidats 2-opt restreints aux k plus proches voisins.

    Un mouvement (i, j) est retenu si au moins une des deux nouvelles arêtes
    (a, c) ou (b, d) relie une ville à l'un de ses k-NN, avec :
        a = tour[i - 1], b = tour[i], c = tour[j], d = tour[(j + 1) % n]

    Cela revient à parcourir :
      - pour chaque position i, les voisins c de a  ->  j = position[c]
      - pour chaque position i, les voisins d de b  ->  j = position[d] - 1

    On déduplique via l'ensemble des couples (i, j) déjà produits, puis on ne
    conserve que les mouvements à gain strictement positif.
    """
    n = len(tour)

    position = [0] * n
    for pos, city in enumerate(tour):
        position[city] = pos

    couples_vus: set[tuple[int, int]] = set()
    candidats: list[TwoOptCandidat] = []
    next_id = 0

    def ajouter(i: int, j: int) -> None:
        nonlocal next_id
        if j <= i or j >= n:
            return
        if (i, j) in couples_vus:
            return
        couples_vus.add((i, j))
        gain = calcul_gain_2opt(instance, tour, i, j)
        if gain > DELTA_EPS:
            candidats.append({"id": next_id, "indices": (i, j), "gain": gain})
            next_id += 1

    for i in range(1, n - 1):
        a = tour[i - 1]
        b = tour[i]
        for c in knn[a]:
            ajouter(i, position[c])
        for d in knn[b]:
            ajouter(i, position[d] - 1)

    return candidats


def solve_2opt_lot_knn(
    instance: TSPInstance,
    initial_tour: Tour,
    k: int,
    knn: list[list[int]] | None = None,
    verbose: bool = True,
) -> tuple[Tour, float]:
    """
    Variante "lot compatible" + k-NN : on combine les deux idées du TER.

    1. on ne génère les candidats 2-opt que parmi les paires autorisées par
       la liste k-NN (étape 2) ;
    2. on construit le graphe de conflits, on sélectionne un MIS glouton,
       puis on applique le lot simultanément (étape 1) ;
    3. on répète jusqu'à saturation.
    """
    tour: Tour = list(initial_tour)
    if knn is None:
        knn = compute_knn(instance, k)

    best_cost = calculate_tour_cost(instance, tour)
    iteration = 0

    if verbose:
        print(
            f"Début 2-opt lot + k-NN (k={k})... coût initial : {best_cost:.2f}"
        )

    while True:
        iteration += 1
        candidats = generer_candidats_knn(instance, tour, knn)
        if not candidats:
            break

        incompatibles = construire_graphe_conflits(candidats)
        lot = selectionner_lot_glouton(candidats, incompatibles)
        if not lot:
            break

        ancien_cout = best_cost
        gain_theorique = sum(c["gain"] for c in lot)
        tour = appliquer_lot_simultanement(tour, lot)
        best_cost = calculate_tour_cost(instance, tour)
        gain_reel = ancien_cout - best_cost

        if verbose:
            print(
                f"Itération {iteration}: "
                f"{len(candidats)} candidats k-NN, "
                f"{len(lot)} mouvements appliqués, "
                f"gain théorique {gain_theorique:.2f}, "
                f"gain réel {gain_reel:.2f}, "
                f"coût {best_cost:.2f}"
            )

        if gain_reel <= DELTA_EPS:
            break

    if verbose:
        print(f"Fin 2-opt lot + k-NN. coût final : {best_cost:.2f}")

    return tour, best_cost


def benchmark_knn(
    instance: TSPInstance,
    initial_tour: Tour,
    ks: list[int],
    verbose: bool = True,
) -> list[dict]:
    """
    Exécute solve_2opt_lot_knn pour chaque valeur de k et collecte, pour chacune,
    le coût final et le temps de résolution (en secondes).
    """
    results: list[dict] = []
    for k in ks:
        knn = compute_knn(instance, k)
        t0 = time.perf_counter()
        _, cost = solve_2opt_lot_knn(instance, initial_tour, k, knn=knn, verbose=False)
        elapsed = time.perf_counter() - t0
        results.append({"k": k, "cost": cost, "time": elapsed})
        if verbose:
            print(f"k={k:>3} | coût={cost:.2f} | temps={elapsed * 1000:.1f} ms")
    return results


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(base_dir, "data", "berlin52.tsp")

    instance = TSPInstance(file_path)
    initial_tour, initial_cost = solve_nn(instance, start_node=0)
    print(f"Coût NN initial : {initial_cost:.2f}")

    # Référence : 2-opt complet pour comparer qualité et temps
    t0 = time.perf_counter()
    _, full_cost = solve_2opt(instance, initial_tour)
    full_time = time.perf_counter() - t0
    print(f"2-opt complet : coût={full_cost:.2f}, temps={full_time * 1000:.1f} ms")

    # Référence : 2-opt lot (sans k-NN) pour comparer à stratégie batch égale
    t0 = time.perf_counter()
    _, lot_full_cost = solve_2opt_lot(instance, initial_tour, verbose=False)
    lot_full_time = time.perf_counter() - t0
    print(
        f"2-opt lot (sans k-NN) : coût={lot_full_cost:.2f}, temps={lot_full_time * 1000:.1f} ms"
    )

    knn_demo = compute_knn(instance, k=10)
    t0 = time.perf_counter()
    lot_tour, lot_cost = solve_2opt_lot_knn(
        instance, initial_tour, k=10, knn=knn_demo, verbose=True
    )
    lot_time = time.perf_counter() - t0
    print(f"2-opt lot + k-NN : coût={lot_cost:.2f}, temps={lot_time * 1000:.1f} ms")

    candidats_knn = generer_candidats_knn(instance, initial_tour, knn_demo)
    graphe_knn = construire_graphe_conflits(candidats_knn)
    lot_knn = selectionner_lot_glouton(candidats_knn, graphe_knn)
    plot_conflict_graph(
        candidats_knn,
        graphe_knn,
        title=(
            f"Graphe de conflits 2-opt k-NN (k=10) — {instance.name} "
            "(tour NN initial)"
        ),
        save_path=os.path.join(base_dir, "results", "conflict_graph_knn.png"),
        highlight_ids={c["id"] for c in lot_knn},
    )
    plot_tour(
        instance,
        lot_tour,
        title=f"2-opt lot + k-NN (k=10) — {instance.name}",
        save_path=os.path.join(base_dir, "results", "tour_2opt_lot_knn_k10.png"),
    )

    # Étude du compromis k vs temps vs qualité (2-opt lot + k-NN)
    ks = [2, 5, 10, 15, 20, 30, 40, instance.dimension - 1]
    results = benchmark_knn(instance, initial_tour, ks)

    plot_knn_tradeoff(
        results,
        reference=[
            {"cost": full_cost, "time": full_time, "label": "2-opt complet"},
            {"cost": lot_full_cost, "time": lot_full_time, "label": "2-opt lot"},
        ],
        title=f"Compromis k vs temps vs qualité (2-opt lot + k-NN) — {instance.name}",
        save_path=os.path.join(base_dir, "results", "knn_tradeoff.png"),
    )
