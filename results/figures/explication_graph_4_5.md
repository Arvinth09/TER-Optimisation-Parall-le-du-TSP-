
## Graphique 2 : La Qualité (Gap %) - `2_quality.png`

**Le "curseur noir" (Barre d'erreur)** : 
- Il représente l'**Écart-type** (la variation entre tes 5 runs).
- Le sommet de la barre colorée est la **moyenne**.
- Le trait noir indique la **stabilité** : plus il est petit, plus l'algorithme est fiable et redonne le même score à chaque essai. Un grand trait signale que l'algorithme est plus dépendant du hasard.

---

## Graphique 4 : Le Compromis (Pareto) - `4_pareto.png`

**C'est quoi ?** C'est le graphique "Rapport Qualité/Vitesse".
- **Axe X (Temps)** : Plus on est à gauche, plus c'est rapide.
- **Axe Y (Gap)** : Plus on est bas, plus le trajet est court (précis).
- **Analyse** : Le VNS gagne en précision (plus bas) mais perd en temps (plus à droite) par rapport au 2-opt. C'est l'arbitrage classique en optimisation.

---

## Graphique 5 : Focus VNS (Boîtes à moustaches / Boxplots) - `5_vns_focus.png`

**Pourquoi seulement 2 algorithmes ?** 
Nous avons choisi de ne comparer que le **2-opt** (la référence) et le **VNS** (notre meilleure méthode) sur ce graphique. 
*   *Raison technique* : Si on ajoutait le NN (22% de gap), l'échelle du graphique serait tellement grande que les boîtes du 2-opt et du VNS paraîtraient minuscules et illisibles.
*   *Objectif* : Braquer les projecteurs sur le "duel" final entre la recherche locale de base et notre métaheuristique hybride.

**Comment lire la boîte (Boxplot) ?**
1.  **La barre au milieu** : C'est le résultat "médian". Plus elle est basse, plus l'algo est performant.
2.  **La taille de la boîte** : Elle montre la **robustesse**. Une boîte ultra-plate signifie que l'algorithme est virtuellement infaillible sur ces instances.
3.  **Les moustaches** : Elles indiquent le meilleur et le pire score obtenus sur les 5 essais.
