import sys
import os

# Ajouter le dossier src au chemin de recherche de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from instance import TSPInstance
except ImportError:
    # Cas où on lance depuis la racine du projet
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    from instance import TSPInstance

def test_berlin52():
    print("--- Test du Parser sur berlin52.tsp ---")
    # Recherche du fichier data/berlin52.tsp
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_dir, 'data', 'berlin52.tsp')
    
    if not os.path.exists(file_path):
        print(f"Erreur : le fichier {file_path} n'existe pas.")
        return

    try:
        instance = TSPInstance(file_path)
        print(f"Succès ! {instance}")
        print(f"Dimension : {instance.dimension}")
        print(f"Première ville (ID 1) : {instance.coords[0]}")
        print(f"Dernière ville (ID 52) : {instance.coords[51]}")
        
        # Test calcul de distance entre ville 1 et 2 (Euclidienne)
        # Ville 1 : (565, 575), Ville 2 : (25, 185)
        # d = sqrt((565-25)^2 + (575-185)^2) = sqrt(540^2 + 390^2) = sqrt(291600 + 152100) = sqrt(443700) approx 666.11
        d = instance.get_distance(0, 1)
        print(f"Distance entre ville 1 et 2 : {d:.2f}")
        
    except Exception as e:
        print(f"Erreur lors du test : {e}")

if __name__ == "__main__":
    test_berlin52()
