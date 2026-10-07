import os
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx

from instance import TSPInstance
from solver import Tour, calculate_tour_cost


def plot_conflict_graph(
    candidats: Sequence[Mapping[str, Any]],
    incompatibles: dict[int, list[int]],
    title: str = "Graphe de conflits (2-opt)",
    save_path: str | None = None,
    highlight_ids: set[int] | None = None,
    show: bool = True,
) -> None:
    """
    Affiche le graphe d'incompatibilité entre candidats 2-opt.
    Layout circulaire, taille des sommets proportionnelle au gain.
    highlight_ids : sommets à mettre en évidence (ex. lot MIS), en vert.
    """
    n = len(candidats)
    if n == 0:
        print("Aucun candidat : graphe vide, rien à afficher.")
        return

    highlight_ids = highlight_ids or set()

    graph = nx.Graph()
    for c in candidats:
        graph.add_node(int(c["id"]))

    for u, voisins in incompatibles.items():
        for v in voisins:
            graph.add_edge(u, v)

    pos = nx.circular_layout(graph, scale=1.0)
    for node in pos:
        x, y = pos[node]
        pos[node] = (y, -x)  # début du cercle en haut de la figure

    gains = [float(c["gain"]) for c in candidats]
    g_max = max(gains) if gains else 1.0

    nodelist = [int(c["id"]) for c in candidats]
    node_sizes = [80 + 220 * (float(c["gain"]) / g_max) for c in candidats]
    node_colors = [
        "#2ca02c" if int(c["id"]) in highlight_ids else "#1f77b4" for c in candidats
    ]
    labels = {
        int(c["id"]): f"{c['id']}\n({c['indices'][0]},{c['indices'][1]})"
        for c in candidats
    }

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="0.65",
        alpha=0.55,
        width=0.9,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=nodelist,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.6,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels=labels,
        font_size=7,
        font_color="black",
    )

    ax.set_aspect("equal")
    ax.axis("off")
    m_edges = graph.number_of_edges()
    ax.set_title(f"{title}\n{len(candidats)} sommets, {m_edges} arêtes")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"Graphe de conflits sauvegardé dans : {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_knn_tradeoff(
    results: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    title: str = "Compromis k vs temps vs qualité",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Trace la courbe coût final et temps d'exécution en fonction de k.

    - `results` : liste de dicts {"k", "cost", "time"} (temps en secondes).
    - `reference` (optionnel) : une référence ou une liste de références,
      chaque dict au format {"cost", "time", "label"}.
    """
    if not results:
        print("Aucun résultat : rien à tracer.")
        return

    ks = [int(r["k"]) for r in results]
    costs = [float(r["cost"]) for r in results]
    times = [float(r["time"]) for r in results]

    cost_color = "#1f77b4"
    time_color = "#d62728"

    fig, ax_cost = plt.subplots(figsize=(9, 6))
    ax_cost.plot(ks, costs, "o-", color=cost_color, label="coût final")
    ax_cost.set_xlabel("k (plus proches voisins)")
    ax_cost.set_ylabel("Coût du tour", color=cost_color)
    ax_cost.tick_params(axis="y", labelcolor=cost_color)
    ax_cost.grid(alpha=0.3)

    ax_time = ax_cost.twinx()
    ax_time.plot(ks, times, "s--", color=time_color, label="temps (s)")
    ax_time.set_ylabel("Temps d'exécution (s)", color=time_color)
    ax_time.tick_params(axis="y", labelcolor=time_color)

    if reference is not None:
        references: Sequence[Mapping[str, Any]]
        if isinstance(reference, Mapping):
            references = [reference]
        else:
            references = reference

        ref_styles = [":", "-.", (0, (1, 1)), (0, (3, 2, 1, 2))]
        for idx, ref in enumerate(references):
            label = str(ref.get("label", f"référence {idx + 1}"))
            linestyle = ref_styles[idx % len(ref_styles)]
            if "cost" in ref:
                ax_cost.axhline(
                    float(ref["cost"]),
                    color=cost_color,
                    linestyle=linestyle,
                    alpha=0.7,
                    label=f"{label} (coût)",
                )
            if "time" in ref:
                ax_time.axhline(
                    float(ref["time"]),
                    color=time_color,
                    linestyle=linestyle,
                    alpha=0.7,
                    label=f"{label} (temps)",
                )

    lines_cost, labels_cost = ax_cost.get_legend_handles_labels()
    lines_time, labels_time = ax_time.get_legend_handles_labels()
    ax_cost.legend(
        lines_cost + lines_time,
        labels_cost + labels_time,
        loc="upper left",
        bbox_to_anchor=(1.12, 1.0),
        borderaxespad=0.0,
    )

    plt.title(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"Graphique compromis sauvegardé dans : {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_tour(
    instance: TSPInstance,
    tour: Tour,
    title: str = "TSP Tour",
    save_path: str | None = None,
) -> None:
    coords = instance.coords
    assert coords is not None

    ordered_coords = coords[tour + [tour[0]]]

    plt.figure(figsize=(10, 7))
    plt.plot(
        ordered_coords[:, 0],
        ordered_coords[:, 1],
        "b-",
        alpha=0.6,
        linewidth=1,
        label="Trajet",
    )

    plt.scatter(coords[:, 0], coords[:, 1], c="red", s=40, zorder=5, label="Villes")
    plt.scatter(
        coords[tour[0], 0],
        coords[tour[0], 1],
        c="green",
        s=100,
        marker="*",
        zorder=6,
        label="Départ",
    )

    cost = calculate_tour_cost(instance, tour)

    plt.title(f"{title}\nCoût: {cost:.2f}")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")

    if save_path:
        plt.savefig(save_path)
        print(f"Graphique sauvegardé dans : {save_path}")

    plt.show()
