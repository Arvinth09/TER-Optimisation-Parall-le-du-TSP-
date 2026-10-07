"""
Module 3 : Visualisation des performances (Phase 4).
V2 : Ajout Graphique Temps vs Algo et Sensibilité à K.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
RAW_CSV = "results/phase4_raw.csv"
OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-muted')

def generate_graphs():
    if not os.path.exists(RAW_CSV):
        print(f"Erreur : {RAW_CSV} non trouvé.")
        return
    
    df = pd.read_csv(RAW_CSV)
    instances = df['instance'].unique()
    
    # Détection automatique des noms (pour compatibilité ancien/nouveau run)
    available_algos = df['algo'].unique()
    
    # On cherche les algos qui nous intéressent par mot-clé
    def get_algo_name(keyword):
        for name in available_algos:
            if keyword in name: return name
        return None

    algo_nn = get_algo_name("NN")
    algo_2opt = get_algo_name("2-opt") # Prendra soit '2-opt K-NN' soit '2-opt lot + K-NN'
    algo_3opt = get_algo_name("3-opt Par (4 p)")
    algo_vns = get_algo_name("VNS")

    algos_select = [n for n in [algo_nn, algo_2opt, algo_3opt, algo_vns] if n is not None]
    
    # --- 1. SPEEDUP ---
    plt.figure(figsize=(10, 6))
    df_par = df[df['algo'].str.contains('3-opt Par')].copy()
    for instance in instances:
        inst_par = df_par[df_par['instance'] == instance]
        t1 = inst_par[inst_par['procs'] == 1]['time_s'].mean()
        procs_list = sorted(inst_par['procs'].unique())
        speedups = [t1 / inst_par[inst_par['procs'] == p]['time_s'].mean() for p in procs_list]
        plt.plot(procs_list, speedups, 'o-', label=f'Instance {instance}')
    plt.plot([1, 8], [1, 8], 'k--', alpha=0.3, label='Idéal (S=N)')
    plt.title("Graphique 1 : Speedup vs Nb processus (3rd-opt Parallel)")
    plt.xlabel("Nombre de processus")
    plt.ylabel("Speedup (Relatif à 1 proc)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/1_speedup.png", dpi=150)

    # --- 2. QUALITE (GAP %) ---
    plt.figure(figsize=(12, 6))
    df_q = df[df['algo'].isin(algos_select)]
    summary_q = df_q.groupby(['instance', 'algo'])['gap_percent'].agg(['mean', 'std']).unstack()
    summary_q['mean'].plot(kind='bar', yerr=summary_q['std'], capsize=4, ax=plt.gca())
    plt.title("Graphique 2 : Qualité (Gap %) par algorithme")
    plt.ylabel("Gap à l'optimum (%)"); plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/2_quality.png", dpi=150)

    # --- 3. TEMPS VS ALGO (NOUVEAU) ---
    plt.figure(figsize=(12, 6))
    summary_t = df_q.groupby(['instance', 'algo'])['time_s'].mean().unstack()
    summary_t.plot(kind='bar', ax=plt.gca())
    plt.yscale('log') # Log scale car l'écart NN vs VNS est massif
    plt.title("Graphique 3 : Temps d'exécution par algorithme (Echelle Log)")
    plt.ylabel("Temps moyen (s) - Log Scale"); plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/3_time_per_algo.png", dpi=150)

    # --- 4. COMPROMIS PARETO ---
    plt.figure(figsize=(10, 6))
    for algo in algos_select:
        df_algo = df[df['algo'] == algo]
        sum_a = df_algo.groupby('instance').agg({'time_s': 'mean', 'gap_percent': 'mean'}).sort_values('time_s')
        plt.plot(sum_a['time_s'], sum_a['gap_percent'], 'o-', label=algo)
    plt.xscale('log'); plt.title("Graphique 4 : Compromis Temps vs Qualité"); plt.xlabel("Temps (s)"); plt.ylabel("Gap (%)"); plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(f"{OUTPUT_DIR}/4_pareto.png", dpi=150)

    # --- 5. FOCUS VNS ROBUSTESSE ---
    inst_large = "pr1002"
    if inst_large in instances:
        plt.figure(figsize=(8, 6))
        data = [df[(df['instance'] == inst_large) & (df['algo'] == a)]['gap_percent'] for a in ["2-opt K-NN", "VNS (Hybrid)"]]
        plt.boxplot(data, labels=["2-opt K-NN", "VNS (Hybrid)"])
        plt.title(f"Graphique 5 : Stabilité du VNS sur {inst_large}")
        plt.ylabel("Gap (%)"); plt.grid(axis='y', alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/5_vns_focus.png", dpi=150)

    print(f"Tous les graphiques ont été générés dans {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_graphs()
