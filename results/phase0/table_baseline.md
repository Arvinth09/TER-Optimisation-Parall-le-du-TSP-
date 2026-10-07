# Performance des Algorithmes de Base (Phase 0)
**Instance : Berlin52 (Optimum de référence : 7542)**

| Algorithme | Distance | Écart (Gap %) | Temps (ms) |
| :--- | :--- | :--- | :--- |
| Nearest Neighbor (Départ 0) | 8981 | +19,08% | 5,04 ms |
| Best Nearest Neighbor (Tous départs) | 8182 | +8,49% | 181,49 ms |
| 2-opt Séquentiel (Baseline) | 7670 | +1,69% | 96,71 ms |
| **2-opt Lot + K-NN (Phase 1&2)** | **7841** | **+3,97%** | **42,81 ms** |

### Justifications et Analyse (Pour le rapport) :
- **Efficacité de la Recherche Locale** : Le passage du Best NN au 2-opt montre un gain significatif de qualité (**8182 → 7670**), prouvant que la recherche locale est indispensable pour s'approcher de l'optimum.
- **Vitesse vs Précision (Le compromis K-NN)** : On remarque que la version **Lot + K-NN** est plus rapide (42 ms), mais légèrement moins précise (+3,97%) que le 2-opt complet. C'est normal : en restreignant la recherche aux 20 voisins les plus proches, on gagne en temps ce qu'on perd marginalement en exploration exhaustive.
- **Speedup Algorithmique** : Sur une instance de taille moyenne comme `ch150`, l'optimisation Batch + K-NN offre un **Speedup de x12,1** par rapport au séquentiel (0,29s contre 3,51s). Cet avantage s'accentue exponentiellement avec la taille des instances.
- **Transition** : Cette accélération massive justifie le besoin de parallélisation et d'optimisations algorithmiques (K-NN, Graphe de conflits) qui seront abordés dans les phases suivantes.
