"""
Script pour générer des figures individuelles aérées NN vs VNS.
Permet d'avoir une meilleure lisibilité sur les instances denses comme pcb442.
"""

import os
import matplotlib.pyplot as plt
from instance import TSPInstance
from solver import solve_nn
from vns_solver import solve_vns

def save_individual_plot(instance, tour, title, filename, color, linewidth, marker_size):
    coords = instance.coords
    assert coords is not None
    
    plt.figure(figsize=(12, 9))
    ordered_coords = coords[tour + [tour[0]]]
    
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], color=color, alpha=0.7, linewidth=linewidth, label="Trajet")
    plt.scatter(coords[:, 0], coords[:, 1], c="black", s=marker_size, zorder=5)
    plt.scatter(coords[tour[0], 0], coords[tour[0], 1], c="green", s=marker_size*4, marker="*", zorder=6, label="Départ")
    
    plt.title(title, fontsize=18)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=12)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Image sauvegardée : {filename}")

def process_instance(instance_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inst_path = os.path.join(base_dir, "data", f"{instance_name}.tsp")
    
    print(f"\n--- Traitement Instance {instance_name} ---")
    instance = TSPInstance(inst_path)
    
    # 1. Calcul NN
    tour_nn, cost_nn = solve_nn(instance, start_node=0)
    
    # 2. Calcul VNS
    tour_vns, cost_vns = solve_vns(instance, tour_nn, k_neighbors=20, max_kicks=10, verbose=False)
    
    # 3. Sauvegarde Individuelle
    out_nn = os.path.join(base_dir, "results", "phase4", f"viz_{instance_name}_1_NN.png")
    out_vns = os.path.join(base_dir, "results", "phase4", f"viz_{instance_name}_2_VNS.png")
    
    # Paramètres de style adaptés à la densité
    lw = 1.0 if instance.dimension > 100 else 1.5
    ms = 15 if instance.dimension > 100 else 30
    
    save_individual_plot(instance, tour_nn, f"Tour Initial (NN) - {instance_name}\nCoût: {cost_nn:.2f}", out_nn, "red", lw, ms)
    save_individual_plot(instance, tour_vns, f"Tour Optimisé (VNS) - {instance_name}\nCoût: {cost_vns:.2f}", out_vns, "blue", lw, ms)

if __name__ == "__main__":
    process_instance("berlin52")
    process_instance("pcb442")
