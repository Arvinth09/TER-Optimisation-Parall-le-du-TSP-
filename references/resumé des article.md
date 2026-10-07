
### 1. Les trois références : Titres et Résumés

#### **Référence [1]**
*   **Titre exact :** *A parallel 2-opt algorithm for the traveling salesman problem* (1995)
*   **Auteurs :** M.G.A. Verhoeven, E.H.L. Aarts, P.C.J. Swinkels.
*   **Résumé :** Cet article est une étude **pratique**. Il décrit comment paralléliser l'algorithme 2-opt sur une machine à 512 processeurs. Les auteurs utilisent le concept de "parallélisme de données" en divisant le trajet en segments distribués. Ils prouvent qu'une recherche locale parallèle peut atteindre la même qualité de solution qu'une recherche séquentielle, mais beaucoup plus rapidement (accélération/speedup).

#### **Référence [2]**
*   **Titre exact :** *Parallel local search* (1995)
*   **Auteurs :** M.G.A. Verhoeven, E.H.L. Aarts.
*   **Résumé :** Cet article est une étude **théorique et classificatoire** (une "survey"). Il dresse une taxonomie complète des stratégies de recherche locale parallèle. Il définit des concepts fondamentaux comme le "Single-walk" (un seul chemin exploré) contre le "Multiple-walk" (plusieurs chemins simultanés), ainsi que les notions de synchronisation. C'est la base méthodologique pour toute personne travaillant sur le TSP parallèle.

#### **Référence [3]**
*   **Titre exact :** *Breaking TSP local search barriers: scalable multi-GPU parallelisation of 4-opt, 5-opt, 6-opt and hybrid variable λ-opt* (2026)
*   **Auteurs :** Wen-Bao Qiao, Jean-Charles Créput, Nan Wang, Kun Meng, Abdelkhalek Mansouri.
*   **Résumé :** Cet article représente l'**état de l'art actuel** (très récent). Il démontre qu'il est possible de briser la "barrière" du 3-opt pour aller vers le 4-opt, 5-opt et 6-opt grâce à la puissance des GPU. Il explique comment gérer l'explosion combinatoire des schémas de reconnexion et utilise des stratégies de voisinage variable ($\lambda$-opt) et des listes de candidats (K-NN) pour rester efficace sur des instances massives.

---

### 2. Quel article as-tu utilisé pour quel point de ton projet ?

Voici comment tu peux "mapper" ton travail sur ces références dans ton rapport :
---

| Étape du Projet | Référence | Localisation & Citation | Pourquoi ce choix ? |
| :--- | :--- | :--- | :--- |
| **Phase 1 : Graphe de conflits** | **Ref [1]** | **P. 3, Sec. 3.1 :** *« The idea of tailored data parallelism is to partition a solution into a number of disjoint partial solutions. »* | Justifie l'utilisation du graphe de conflits pour identifier des mouvements "disjoints" applicables simultanément. |
| **Phase 2 : Réduction K-NN** | **Ref [3]** | **P. 2, Sec. 2.2 :** *« LKH considers nearest neighbour information [...] to reduce the k-opt search space. »* | Prouve que limiter la recherche aux $k$ voisins (4, 5, 6-opt) est **obligatoire** pour la performance. |
| **Phase 3 : 3-opt Parallèle** | **Ref [2]** | **P. 6, Sec. 3.2 :** *« The idea of single-step parallelism is to evaluate neighbors simultaneously and subsequently make a single step. »* | Définit ta stratégie : (Single-step, Parallel-step" ) les processus CPU explorent des morceaux de voisinage en parallèle pour un seul tour. |
| **Phase 3.5 : k-opt & Double-Bridge** | **Ref [3]** | **P. 2, Sec. 2.1 :** *« The 3-opt, 4-opt, 5-opt, 6-opt [...] operations follow analogous principles to 2-opt. »* **P. 4, Sec. 4.1 :** *« The fundamental challenge lies in the $2^{k-1} \times (k-1)!$ possible reconnection schemes. »* | Explique pourquoi on limites le 4-opt au Double-Bridge : l'explosion combinatoire (1935 schémas pour $k=6$) rend l'exhaustivité impossible sur CPU. |
| **Architecture (VNS / Orchestrateur)** | **Ref [3]** | **P. 2, Sec. 2.2 :** *« At each iteration step the algorithm examines, for ascending values of k, whether an interchange [...] results in a shorter tour. »* | Le passage de 2-opt à 3-opt puis 4-opt est la définition même du "Variable Neighborhood Search" (Ref 2) et du "Hybrid Variable $\lambda$-opt" (Ref 3). |
---




