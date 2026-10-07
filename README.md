# Optimisation parallèle du TSP

##  Présentation

Ce projet a été réalisé dans le cadre d'un **Travail d'Étude et de Recherche (TER)** à l'Université d'Évry – Paris-Saclay.

L'objectif est d'étudier différentes stratégies pour **accélérer la recherche locale appliquée au problème du voyageur de commerce (TSP)**, tout en conservant une bonne qualité de solution.

Le TSP consiste à trouver un circuit de coût minimal passant une seule fois par chaque ville et revenant au point de départ. Les méthodes exactes devenant rapidement trop coûteuses sur de grandes instances, nous nous sommes intéressés à des **méthodes heuristiques et parallèles**.

---

## Problématique

> **Comment paralléliser et accélérer la recherche locale du TSP sur de grandes instances tout en conservant une bonne qualité de solution ?**

Notre approche repose principalement sur les mouvements **2-opt et 3-opt**, la réduction du voisinage grâce aux **k plus proches voisins (k-NN)** et différentes stratégies de parallélisation.

---

## Approches étudiées

### 1. Recherche locale 2-opt

Le mouvement 2-opt consiste à supprimer deux arêtes d'une solution et à reconnecter les extrémités afin de supprimer les croisements et réduire la longueur totale du circuit.

Cette méthode permet d'améliorer rapidement une solution initiale, mais l'exploration exhaustive de tous les mouvements 2-opt peut devenir coûteuse.

### 2. 2-opt avec k-NN

Afin de réduire le nombre de mouvements à explorer, nous limitons la recherche aux **k plus proches voisins** de chaque ville.

Cette stratégie permet de passer d'un voisinage potentiellement quadratique à un voisinage dépendant de `k`, avec une complexité approximative de :

```text
O(n × k)
