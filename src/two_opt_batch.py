"""
Travail (1) du TER : génération de candidats 2-opt, graphe de conflits,
sélection gloutonne d'un ensemble indépendant maximal, puis application
simultanée des mouvements compatibles.

Module : two_opt_batch (2-opt par lots compatibles).
"""

import os
from typing import TypedDict

from instance import TSPInstance
from solver import Tour, calculate_tour_cost, solve_nn, two_opt_swap
from visualizer import plot_conflict_graph

DELTA_EPS = 1e-5


class TwoOptCandidat(TypedDict):
    id: int
    indices: tuple[int, int]
    gain: float


def calcul_gain_2opt(instance: TSPInstance, tour: Tour, i: int, j: int) -> float:
    """
    Gain théorique du mouvement 2-opt (i, j).
    On ne recalcule que les 4 arêtes touchées.
    """
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]

    current_dist = instance.get_distance(a, b) + instance.get_distance(c, d)
    new_dist = instance.get_distance(a, c) + instance.get_distance(b, d)
    return current_dist - new_dist


def generer_candidats(instance: TSPInstance, tour: Tour) -> list[TwoOptCandidat]:
    """
    Génère tous les mouvements 2-opt à gain strictement positif.

    Chaque candidat contient :
    - id : identifiant unique
    - indices : tuple (i, j)
    - gain : gain théorique du mouvement
    """
    candidats: list[TwoOptCandidat] = []
    n = len(tour)
    next_id = 0

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            gain = calcul_gain_2opt(instance, tour, i, j)
            if gain > DELTA_EPS:
                candidats.append(
                    {
                        "id": next_id,
                        "indices": (i, j),
                        "gain": gain,
                    }
                )
                next_id += 1

    return candidats


def mouvements_en_conflit(candidat_a: TwoOptCandidat, candidat_b: TwoOptCandidat) -> bool:
    """
    Deux mouvements 2-opt sont incompatibles s'ils :
    - se chevauchent sur le segment inversé
    - ou se touchent, ce qui revient à partager une arête coupée

    Compatibles <=> leurs segments [i, j] sont séparés par au moins
    une arête intacte sur le tour linéarisé.
    """
    i1, j1 = candidat_a["indices"]
    i2, j2 = candidat_b["indices"]

    if i1 > i2:
        i1, j1, i2, j2 = i2, j2, i1, j1

    return j1 + 1 >= i2


def construire_graphe_conflits(
    candidats: list[TwoOptCandidat],
) -> dict[int, list[int]]:
    """
    Graphe de conflits :
    - un sommet = un mouvement 2-opt candidat
    - une arête = incompatibilité entre deux mouvements
    """
    incompatibles: dict[int, list[int]] = {c["id"]: [] for c in candidats}

    for index_a in range(len(candidats)):
        for index_b in range(index_a + 1, len(candidats)):
            candidat_a = candidats[index_a]
            candidat_b = candidats[index_b]

            if mouvements_en_conflit(candidat_a, candidat_b):
                id_a = candidat_a["id"]
                id_b = candidat_b["id"]
                incompatibles[id_a].append(id_b)
                incompatibles[id_b].append(id_a)

    return incompatibles


def selectionner_lot_glouton(
    candidats: list[TwoOptCandidat], incompatibles: dict[int, list[int]]
) -> list[TwoOptCandidat]:
    """
    Heuristique MIS gloutonne pondérée par le gain :
    on trie les candidats par gain décroissant et on conserve
    uniquement ceux qui ne sont en conflit avec aucun déjà choisi.
    """
    lot: list[TwoOptCandidat] = []
    interdits: set[int] = set()

    for candidat in sorted(candidats, key=lambda x: x["gain"], reverse=True):
        mon_id = candidat["id"]
        if mon_id in interdits:
            continue

        lot.append(candidat)
        interdits.add(mon_id)
        interdits.update(incompatibles.get(mon_id, []))

    return lot


def appliquer_lot_simultanement(tour: Tour, lot: list[TwoOptCandidat]) -> Tour:
    """
    Applique tous les mouvements compatibles du lot.

    Comme les segments choisis sont disjoints, appliquer les inversions
    de la droite vers la gauche produit le même résultat qu'une
    application simultanée sur le tour courant.
    """
    nouveau_tour: Tour = list(tour)

    for candidat in sorted(lot, key=lambda x: x["indices"][0], reverse=True):
        i, j = candidat["indices"]
        nouveau_tour = two_opt_swap(nouveau_tour, i, j)

    return nouveau_tour


def solve_2opt_lot(
    instance: TSPInstance, initial_tour: Tour, verbose: bool = True
) -> tuple[Tour, float]:
    """
    Variante "lot compatible" du 2-opt :
    1. générer tous les candidats positifs
    2. construire le graphe de conflits
    3. sélectionner un ensemble indépendant maximal glouton
    4. appliquer simultanément le lot obtenu
    5. répéter jusqu'à ce qu'il n'y ait plus de gain
    """
    tour: Tour = list(initial_tour)
    best_cost = calculate_tour_cost(instance, tour)
    iteration = 0

    if verbose:
        print(
            f"Début de l'optimisation par lots 2-opt... Coût initial : {best_cost:.2f}"
        )

    while True:
        iteration += 1
        candidats = generer_candidats(instance, tour)
        if not candidats:
            break

        incompatibles = construire_graphe_conflits(candidats)
        lot = selectionner_lot_glouton(candidats, incompatibles)
        if not lot:
            break

        ancien_cout = best_cost
        gain_theorique = sum(candidat["gain"] for candidat in lot)
        tour = appliquer_lot_simultanement(tour, lot)
        best_cost = calculate_tour_cost(instance, tour)
        gain_reel = ancien_cout - best_cost

        if verbose:
            print(
                f"Itération {iteration}: "
                f"{len(candidats)} candidats, "
                f"{len(lot)} mouvements appliqués, "
                f"gain théorique {gain_theorique:.2f}, "
                f"gain réel {gain_reel:.2f}, "
                f"coût {best_cost:.2f}"
            )

        if gain_reel <= DELTA_EPS:
            break

    if verbose:
        print(f"Fin de l'optimisation par lots 2-opt. Coût final : {best_cost:.2f}")

    return tour, best_cost


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(base_dir, "data", "berlin52.tsp")

    instance = TSPInstance(file_path)
    initial_tour, initial_cost = solve_nn(instance, start_node=0)

    candidats_demo = generer_candidats(instance, initial_tour)
    graphe_demo = construire_graphe_conflits(candidats_demo)
    lot_demo = selectionner_lot_glouton(candidats_demo, graphe_demo)
    plot_conflict_graph(
        candidats_demo,
        graphe_demo,
        title=f"Graphe de conflits 2-opt — {instance.name} (tour NN initial)",
        save_path=os.path.join(base_dir, "results", "conflict_graph_nn.png"),
        highlight_ids={c["id"] for c in lot_demo},
    )

    print(f"Coût NN initial : {initial_cost:.2f}")
    lot_tour, lot_cost = solve_2opt_lot(instance, initial_tour, verbose=True)
    print(f"Amélioration totale : {initial_cost - lot_cost:.2f}")
    print(
        f"Tour final valide : {len(set(lot_tour)) == len(lot_tour) == instance.dimension}"
    )
