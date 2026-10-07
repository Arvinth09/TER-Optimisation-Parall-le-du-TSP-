import os

import numpy as np


class TSPInstance:
    def __init__(self, filename: str) -> None:
        self.name: str = ""
        self.type: str = ""
        self.dimension: int = 0
        self.edge_weight_type: str = ""
        self.coords: np.ndarray | None = None
        self.dist_matrix: np.ndarray | None = None

        self.load(filename)

    def load(self, filename: str) -> None:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Fichier {filename} non trouvé.")

        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        section: str | None = None
        coords_list: list[list[float]] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("NAME"):
                self.name = line.split(":")[-1].strip()
            elif line.startswith("TYPE"):
                self.type = line.split(":")[-1].strip()
            elif line.startswith("DIMENSION"):
                self.dimension = int(line.split(":")[-1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                self.edge_weight_type = line.split(":")[-1].strip()
            elif line == "NODE_COORD_SECTION":
                section = "COORDS"
                continue
            elif line == "EOF":
                break

            if section == "COORDS":
                parts = line.split()
                if len(parts) >= 3:
                    coords_list.append([float(x) for x in parts[1:]])

        self.coords = np.array(coords_list)
        if len(self.coords) != self.dimension:
            print(
                f"Attention: Dimension lue ({len(self.coords)}) différente "
                f"de la dimension annoncée ({self.dimension})"
            )

    def get_distance(self, i: int, j: int) -> float:
        """Calcul de la distance Euclidienne entre la ville i et j (0-indexed)."""
        assert self.coords is not None
        if self.edge_weight_type == "EUC_2D":
            return float(np.linalg.norm(self.coords[i] - self.coords[j]))
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def __str__(self) -> str:
        return (
            f"Instance TSP: {self.name} | Dimension: {self.dimension} | "
            f"Type: {self.edge_weight_type}"
        )


if __name__ == "__main__":
    print("Test du parser...")
