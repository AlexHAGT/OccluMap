"""Chargement des scans, labels et landmarks (lecture seule sur data/raw)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAW = Path("data/raw")


def load_scan(obj_path):
    """Charge un scan .obj sans réordonner ni fusionner les sommets.

    maintain_order=True est indispensable : sinon trimesh fusionne des sommets
    et les labels par sommet ne correspondent plus au maillage.
    """
    import trimesh
    return trimesh.load(obj_path, process=False, maintain_order=True)


def occlusal_direction(mesh) -> np.ndarray:
    """Vecteur unitaire pointant du côté des couronnes.

    Un scan intra-oral est une surface ouverte : la somme des normales des faces,
    pondérées par leur aire, pointe du côté des dents.
    """
    v = (mesh.face_normals * mesh.area_faces[:, None]).sum(axis=0)
    return v / np.linalg.norm(v)


def find_scan(name: str, raw: Path = RAW) -> Path:
    """Retrouve le .obj d'un scan à partir de son nom, ex. '0140W3ND_lower'."""
    pid, jaw = name.rsplit("_", 1)
    hits = sorted((raw / "teeth3ds").glob(f"*/{jaw}/{pid}/{pid}_{jaw}.obj"))
    if not hits:
        raise FileNotFoundError(f"scan introuvable dans {raw / 'teeth3ds'} : {name}")
    return hits[0]


def find_train_landmarks(name: str, raw: Path = RAW) -> Path | None:
    """Fichier de landmarks d'ENTRAÎNEMENT du scan, ou None.

    Ne cherche jamais dans 3dteethland/test (évaluation finale uniquement).
    """
    pid, jaw = name.rsplit("_", 1)
    f = raw / "3dteethland" / "train" / jaw / pid / f"{pid}_{jaw}__kpt.json"
    return f if f.is_file() else None


def load_landmarks(kpt_path, cls: str = "Cusp") -> np.ndarray:
    """Coordonnées (n, 3) des landmarks d'une classe (Cusp, Mesial, Distal, ...)."""
    kpt_path = Path(kpt_path)
    parts = [p.lower() for p in kpt_path.resolve().parts]
    if "3dteethland" in parts and "test" in parts:
        raise PermissionError("Landmarks de test : évaluation finale uniquement, lecture refusée ici.")
    d = json.loads(kpt_path.read_text())
    pts = [o["coord"] for o in d["objects"] if o["class"] == cls]
    return np.asarray(pts, dtype=float).reshape(-1, 3)
