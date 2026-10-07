"""
Script pour générer une figure comparative NN vs VNS.
Conformément au plan for_me/plan_vns_viz.md.
"""

import os
import time
import matplotlib.pyplot as plt
from instance import TSPInstance
from solver import solve_nn, calculate_tour_cost
from vns_solver import solve_vns

def generate_comparison_figure(instance_path: str, output_path: str):
    # 1. Initialisation
    print(f"Chargement de l'instance : {instance_path}")
    instance = TSPInstance(instance_path)
    
    # Paramètres conformes au plan
    k_neighbors = 20
    max_kicks = 10
    
    # 2. Calculs
    print("Calcul du tour Initial (Nearest Neighbor)...")
    t0_nn = time.perf_counter()
    tour_nn, cost_nn = solve_nn(instance, start_node=0)
    t_nn = time.perf_counter() - t0_nn
    print(f"NN terminé en {t_nn:.4f}s. Coût : {cost_nn:.2f}")
    
    print(f"Calcul du tour Optimisé (VNS) avec k={k_neighbors} et kicks={max_kicks}...")
    t0_vns = time.perf_counter()
    # Le VNS utilise le 3-opt parallèle en interne
    tour_vns, cost_vns = solve_vns(instance, tour_nn, k_neighbors=k_neighbors, max_kicks=max_kicks, verbose=True)
    t_vns = time.perf_counter() - t0_vns
    print(f"VNS terminé en {t_vns:.4f}s. Coût : {cost_vns:.2f}")
    
    # 3. Mise en page Matplotlib
    print("Génération de la figure...")
    coords = instance.coords
    assert coords is not None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # --- Sous-graphique 1 : NN ---
    ordered_nn = coords[tour_nn + [tour_nn[0]]]
    ax1.plot(ordered_nn[:, 0], ordered_nn[:, 1], "r-", alpha=0.5, linewidth=1)
    ax1.scatter(coords[:, 0], coords[:, 1], c="black", s=20, zorder=5)
    ax1.scatter(coords[tour_nn[0], 0], coords[tour_nn[0], 1], c="green", s=80, marker="*", zorder=6)
    ax1.set_title(f"Tour Initial (NN)\nCoût: {cost_nn:.2f}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.2)
    
    # --- Sous-graphique 2 : VNS ---
    ordered_vns = coords[tour_vns + [tour_vns[0]]]
    ax2.plot(ordered_vns[:, 0], ordered_vns[:, 1], "b-", alpha=0.7, linewidth=1.5)
    ax2.scatter(coords[:, 0], coords[:, 1], c="black", s=20, zorder=5)
    ax2.scatter(coords[tour_vns[0], 0], coords[tour_vns[0], 1], c="green", s=80, marker="*", zorder=6)
    ax2.set_title(f"Tour Optimisé (VNS)\nCoût: {cost_vns:.2f}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, alpha=0.2)
    
    plt.suptitle(f"Comparaison Baseline (NN) vs Optimisation (VNS) - {instance.name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 4. Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Figure sauvegardée dans : {output_path}")
    # plt.show() # Pas de show en mode non-interactif

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Instance 1 : Berlin52
    inst1 = os.path.join(base_dir, "data", "berlin52.tsp")
    out1 = os.path.join(base_dir, "results", "phase4", "vns_comparison_berlin52.png")
    # generate_comparison_figure(inst1, out1) # Déjà fait
    
    # Instance 2 : PCB442
    inst2 = os.path.join(base_dir, "data", "pcb442.tsp")
    out2 = os.path.join(base_dir, "results", "phase4", "vns_comparison_pcb442.png")
    
    print("\n--- Génération Figure PCB442 ---")
    generate_comparison_figure(inst2, out2)
