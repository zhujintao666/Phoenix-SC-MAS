# src/policy_table.py
import json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree     # pip install scipy

class RandomLookupPolicy:

    def __init__(self, table_path: str = "src/random120_filled.json"):
        data = json.loads(Path(table_path).read_text(encoding="utf-8"))

        self.X = np.array([
            [
                row["inventory"],
                row["demand"],
                row["backlog"],
                row["assets"]
            ]
            for row in data
        ], dtype=float)

        self.y = np.array([row["order"] for row in data], dtype=float)

        self.tree = cKDTree(self.X)

    def get_order(self, inv, dem, bkl, assets, price):
        inv = float(np.asarray(inv).ravel()[0])
        dem = float(np.asarray(dem).ravel()[0])
        bkl = float(np.asarray(bkl).ravel()[0])

        ast_eff = assets / price if price > 0 else 0
        ast_eff = float(np.asarray(ast_eff).ravel()[0])

        query = np.array([inv, dem, bkl, ast_eff], dtype=float)
        dist, idx = self.tree.query(query, k=1)
        qty = int(round(self.y[idx]))
        max_afford = int(assets // price) if price > 0 else qty
        return max(0, min(qty, max_afford))

LookupPolicy = RandomLookupPolicy



