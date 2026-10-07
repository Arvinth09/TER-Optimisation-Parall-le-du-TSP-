# Optimisation parallèle du TSP

## Présentation

Projet réalisé dans le cadre d'un **Travail d'Étude et de Recherche (TER)** à l'Université d'Évry – Paris-Saclay.

Ce projet porte sur l'optimisation et la **parallélisation de méthodes de recherche locale** appliquées au problème du voyageur de commerce (**TSP – Traveling Salesman Problem**).

L'objectif est d'améliorer le temps d'exécution des algorithmes tout en conservant une bonne qualité de solution.

---

## Objectifs

- Étudier des méthodes de recherche locale pour le TSP.
- Réduire le temps d'exploration du voisinage.
- Exploiter le parallélisme pour accélérer les calculs.
- Comparer les performances et la qualité des solutions obtenues.

---

## Méthodes étudiées

Le projet porte notamment sur :

- **2-opt**
- **2-opt avec k plus proches voisins (k-NN)**
- **3-opt parallèle**
- **Variable Neighborhood Search (VNS)**
- **Double-Bridge**
- Parallélisation basée sur un **graphe de conflits**

---

## Approche

La recherche locale 2-opt permet d'améliorer progressivement une solution du TSP.

Afin de réduire le nombre de mouvements explorés, nous utilisons une approche basée sur les **k plus proches voisins**.

Nous étudions également la **parallélisation des mouvements** afin d'exécuter simultanément des opérations compatibles.

Enfin, une approche hybride basée sur **VNS et Double-Bridge** permet d'explorer davantage l'espace des solutions et de limiter les problèmes liés aux minima locaux.

---

## Résultats

Les expérimentations montrent que :

- la réduction du voisinage avec **k-NN** permet de diminuer le temps de calcul ;
- la parallélisation du **3-opt** permet d'obtenir des gains de performance significatifs ;
- le speedup augmente avec le nombre de processus jusqu'à atteindre les limites liées aux ressources matérielles ;
- les approches hybrides permettent d'améliorer l'exploration de l'espace des solutions.

L'étude met ainsi en évidence le compromis entre **temps d'exécution, parallélisme et qualité de la solution**.

---

## Concepts

- Algorithmique
- Optimisation combinatoire
- Recherche locale
- Métaheuristiques
- Calcul parallèle
- Programmation concurrente
- Graphes
- Analyse de performances

---

## Auteurs

**Detchanamourtty Aravindhan**  
**Vadoud Shayan**  
**Mahesanthan Arvinth**

**Encadrant :** Eric Angel

**Université d'Évry – Paris-Saclay — 2026**

---

## Rapport

Le rapport complet du TER présente en détail les méthodes étudiées, les choix d'implémentation, les expérimentations et l'analyse des résultats.
