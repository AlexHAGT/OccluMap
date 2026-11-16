"""Vérification des données Teeth3DS+ / 3DTeethLand (LECTURE SEULE sur data/raw).

Usage (depuis la racine du projet, avec le venv) :
    python scripts/check_teeth3ds.py                      # inventaire complet
    python scripts/check_teeth3ds.py --patient 0140W3ND   # détail d'un patient + test d'occlusion
    python scripts/check_teeth3ds.py --patient 0140W3ND --show          # vue 3D interactive
    python scripts/check_teeth3ds.py --occlusion-sample 20              # test d'occlusion sur 20 patients
    python scripts/check_teeth3ds.py --patient 0140W3ND --screenshot outputs/checks/vue.png

Ce script ne fait qu'ouvrir des fichiers en lecture. La seule écriture possible
est la capture d'écran (--screenshot), refusée si elle vise data/raw.

Règle du projet : les landmarks de test (3dteethland/test) servent uniquement à
l'évaluation finale. Ici, on ne lit que leurs NOMS de fichiers, jamais leur contenu.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

JAWS = ("upper", "lower")
# <ID>_<upper|lower>            -> scan (.obj) ou labels (.json)
# <ID>_<upper|lower>__kpt       -> landmarks (.json) ; un ou deux underscores tolérés
NAME_RE = re.compile(r"^(?P<id>[A-Za-z0-9]+)_(?P<jaw>upper|lower)(?P<kpt>_+kpt)?$")


def parse_name(path: Path):
    """Renvoie (id_patient, arcade, est_un_fichier_de_landmarks) ou None."""
    m = NAME_RE.match(path.stem)
    if not m:
        return None
    return m["id"], m["jaw"], m["kpt"] is not None


# --------------------------------------------------------------------------- #
# Inventaire
# --------------------------------------------------------------------------- #
def build_index(raw: Path):
    """Parcourt data/raw et indexe tout par clé (id_patient, arcade)."""
    scans = defaultdict(list)    # (id, jaw) -> [(partie, chemin .obj)]
    labels = defaultdict(list)   # (id, jaw) -> [(partie, chemin .json)]
    kpts = {"train": {}, "test": {}}  # (id, jaw) -> chemin _kpt.json
    unknown = []

    t3ds = raw / "teeth3ds"
    for part in sorted(p for p in t3ds.iterdir() if p.is_dir()):
        for f in part.rglob("*"):
            if not f.is_file():
                continue
            parsed = parse_name(f)
            if parsed is None or parsed[2]:
                unknown.append(f)
            elif f.suffix.lower() == ".obj":
                scans[parsed[:2]].append((part.name, f))
            elif f.suffix.lower() == ".json":
                labels[parsed[:2]].append((part.name, f))
            else:
                unknown.append(f)

    for subset in ("train", "test"):
        folder = raw / "3dteethland" / subset
        if not folder.is_dir():
            continue
        for f in folder.rglob("*"):
            if not f.is_file():
                continue
            parsed = parse_name(f)
            if parsed is None or not parsed[2] or f.suffix.lower() != ".json":
                unknown.append(f)
            else:
                kpts[subset][parsed[:2]] = f
    return scans, labels, kpts, unknown


def read_splits(splits_dir: Path):
    """Lit training_/testing_ upper/lower .txt -> {nom: set((id, jaw))}."""
    out = {}
    if not splits_dir.is_dir():
        return out
    for f in sorted(splits_dir.glob("*.txt")):
        keys = set()
        for line in f.read_text().split():
            m = NAME_RE.match(line.strip())
            if m:
                keys.add((m["id"], m["jaw"]))
        out[f.stem] = keys
    return out


def title(text: str):
    print(f"\n=== {text} ===")


def report_inventory(scans, labels, kpts, unknown, splits):
    title("1. Inventaire de data/raw/teeth3ds")
    parts = sorted({p for v in scans.values() for p, _ in v})
    print(f"{'partie':<14}{'obj upper':>10}{'obj lower':>10}{'labels':>8}{'patients':>10}")
    for part in parts:
        keys = [k for k, v in scans.items() if any(p == part for p, _ in v)]
        n_lab = sum(1 for k, v in labels.items() if any(p == part for p, _ in v))
        n_up = sum(1 for k in keys if k[1] == "upper")
        n_lo = sum(1 for k in keys if k[1] == "lower")
        print(f"{part:<14}{n_up:>10}{n_lo:>10}{n_lab:>8}{len({k[0] for k in keys}):>10}")

    n_files = sum(len(v) for v in scans.values())
    dup = {k: v for k, v in scans.items() if len(v) > 1}
    patients = {k[0] for k in scans}
    both = {p for p in patients if (p, "upper") in scans and (p, "lower") in scans}
    labelled = set(labels)
    print(f"\nFichiers .obj : {n_files} ; scans distincts (id, arcade) : {len(scans)}")
    print(f"Scans présents dans plusieurs parties (doublons) : {len(dup)}")
    print(f"Patients distincts : {len(patients)} "
          f"(2 arcades : {len(both)} ; 1 seule arcade : {len(patients) - len(both)})")
    print(f"Scans avec labels : {len(labelled & set(scans))} ; sans labels : {len(set(scans) - labelled)} ; "
          f"labels sans scan : {len(labelled - set(scans))}")
    if unknown:
        print(f"Fichiers au nom inattendu : {len(unknown)} (ex. {unknown[0]})")

    title("2. Contenu de data_part_7")
    p7 = {k for k, v in scans.items() if any(p == "data_part_7" for p, _ in v)}
    others = {k for k, v in scans.items() if any(p != "data_part_7" for p, _ in v)}
    p7_pat = {k[0] for k in p7}
    print(f"Scans : {len(p7)} ({len(p7_pat)} patients) ; avec labels : {len(p7 & labelled)}")
    print(f"Scans aussi présents dans les parties 1 à 6 : {len(p7 & others)}")
    print(f"Patients aussi présents dans les parties 1 à 6 : {len(p7_pat & {k[0] for k in others})}")
    for subset in ("train", "test"):
        print(f"Scans ayant un landmark 3DTeethLand '{subset}' : {len(p7 & set(kpts[subset]))} / {len(p7)}")
    for name, keys in splits.items():
        print(f"Scans listés dans le découpage {name} : {len(p7 & keys)}")

    title("3. Découpage officiel (data/splits)")
    if not splits:
        print("Aucun fichier de découpage trouvé.")
    for name, keys in splits.items():
        missing = keys - set(scans)
        print(f"{name:<16}{len(keys):>5} entrées ; scan introuvable : {len(missing)} ; "
              f"sans labels : {len(keys - labelled)}")
    if splits:
        all_split = set().union(*splits.values())
        labelled_scans = labelled & set(scans)
        print(f"Scans labellisés absents de tout découpage : {len(labelled_scans - all_split)}")
        train = set().union(*(v for k, v in splits.items() if k.startswith("training")))
        test = set().union(*(v for k, v in splits.items() if k.startswith("testing")))
        print(f"Scans à la fois en entraînement et en test : {len(train & test)}")
        mixed = {k[0] for k in train} & {k[0] for k in test}
        print(f"Patients dont une arcade est en entraînement et l'autre en test : {len(mixed)}")

    title("4. Landmarks 3DTeethLand : chaque _kpt.json retrouve-t-il son scan ?")
    for subset in ("train", "test"):
        keys = set(kpts[subset])
        missing = sorted(keys - set(scans))
        where = Counter(p for k in keys & set(scans) for p, _ in scans[k])
        print(f"[{subset}] {len(keys)} fichiers ({sum(k[1] == 'upper' for k in keys)} upper, "
              f"{sum(k[1] == 'lower' for k in keys)} lower), {len({k[0] for k in keys})} patients")
        print(f"    scan .obj trouvé : {len(keys) - len(missing)} / {len(keys)} ; manquants : {len(missing)}"
              + (f" (ex. {missing[:5]})" if missing else ""))
        print(f"    parties contenant ces scans : {dict(sorted(where.items()))}")
        print(f"    scans correspondants avec labels FDI : {len(keys & labelled)} / {len(keys)}")
        for name, sk in splits.items():
            if keys & sk:
                print(f"    dans le découpage {name} : {len(keys & sk)}")
    # Champ interne "key" (nom du .obj) : contrôlé sur train uniquement.
    bad_key, classes = [], Counter()
    for (pid, jaw), f in kpts["train"].items():
        d = json.loads(f.read_text())
        if d.get("key") != f"{pid}_{jaw}.obj":
            bad_key.append(f.name)
        classes.update(o["class"] for o in d.get("objects", []))
    print(f"[train] champ 'key' différent du nom de scan attendu : {len(bad_key)}")
    print(f"[train] classes de landmarks (total) : {dict(classes.most_common())}")
    print(f"Scans présents à la fois dans train et test des landmarks : "
          f"{len(set(kpts['train']) & set(kpts['test']))}")


# --------------------------------------------------------------------------- #
# Détail d'un patient / occlusion
# --------------------------------------------------------------------------- #
def load_scan(obj_path: Path):
    """Charge un scan SANS réordonner ni fusionner les sommets.

    maintain_order=True est indispensable : sinon trimesh fusionne des sommets
    et les labels par sommet ne correspondent plus au maillage.
    """
    import trimesh
    return trimesh.load(obj_path, process=False, maintain_order=True)


def load_labels(json_path: Path):
    d = json.loads(json_path.read_text())
    return np.asarray(d["labels"]), np.asarray(d["instances"])


def open_side_direction(mesh):
    """Direction vers laquelle « regardent » les couronnes.

    Un scan intra-oral est une surface ouverte. La somme des normales des faces,
    pondérées par leur aire, pointe du côté des dents (côté occlusal).
    """
    v = (mesh.face_normals * mesh.area_faces[:, None]).sum(axis=0)
    return v / np.linalg.norm(v)


def occlusion_metrics(scans, labels, pid):
    """Indicateurs chiffrés : les deux arcades sont-elles dans un repère commun ?

    En occlusion réelle, on attend :
      - des couronnes qui se font face (cosinus des directions occlusales proche de -1) ;
      - des dents du haut juste « devant » celles du bas (décalage positif de quelques mm
        le long de la direction occlusale du bas) ;
      - des surfaces dentaires qui se touchent presque (distance minimale proche de 0)
        sans se traverser sur une grande étendue.
    """
    from scipy.spatial import cKDTree
    data = {}
    for jaw in JAWS:
        key = (pid, jaw)
        if key not in scans or key not in labels:
            return None
        mesh = load_scan(scans[key][0][1])
        lab, _ = load_labels(labels[key][0][1])
        if len(lab) != len(mesh.vertices):
            return None
        data[jaw] = (mesh, lab)
    (mu, lu), (ml, ll) = data["upper"], data["lower"]
    tu, tl = mu.vertices[lu > 0], ml.vertices[ll > 0]
    du, dl = open_side_direction(mu), open_side_direction(ml)
    dist, _ = cKDTree(tl).query(tu)
    return {
        "cos": float(du @ dl),
        "offset": float((tu.mean(axis=0) - tl.mean(axis=0)) @ dl),
        "centroid_gap": float(np.linalg.norm(tu.mean(axis=0) - tl.mean(axis=0))),
        "d_min": float(dist.min()),
        "d_med": float(np.median(dist)),
        "near": float((dist < 0.5).mean()),
    }


def report_occlusion_sample(scans, labels, n):
    title(f"Test d'occlusion sur un échantillon de {n} patients")
    candidates = sorted(p for p in {k[0] for k in scans}
                        if all((p, j) in scans and (p, j) in labels for j in JAWS))
    step = max(1, len(candidates) // n)
    sample = candidates[::step][:n]
    print("cos : cosinus entre les directions occlusales des 2 arcades (-1 = face à face, +1 = même sens)")
    print("décal. : position des dents du haut par rapport à celles du bas, le long de la direction")
    print("         occlusale du bas, en mm (positif = le haut est bien en face du bas)")
    print("d_min / d_méd : distance des sommets dentaires du haut aux dents du bas (mm)")
    print(f"\n{'patient':<10}{'cos':>7}{'décal.':>9}{'d_min':>8}{'d_méd':>8}{'% < 0,5 mm':>12}")
    rows = []
    for pid in sample:
        m = occlusion_metrics(scans, labels, pid)
        if m is None:
            print(f"{pid:<10}  (ignoré : labels et sommets incohérents)")
            continue
        rows.append(m)
        print(f"{pid:<10}{m['cos']:>7.2f}{m['offset']:>9.1f}{m['d_min']:>8.2f}{m['d_med']:>8.2f}{100 * m['near']:>11.1f}%")
    if rows:
        cos = np.array([r["cos"] for r in rows])
        print(f"\ncos : min {cos.min():.2f}, médiane {np.median(cos):.2f}, max {cos.max():.2f} "
              f"sur {len(rows)} patients")
        print(f"Patients avec couronnes face à face (cos < -0,7) : {(cos < -0.7).sum()} / {len(rows)}")


def report_patient(scans, labels, kpts, splits, pid):
    title(f"Détail du patient {pid}")
    loaded = {}
    for jaw in JAWS:
        key = (pid, jaw)
        if key not in scans:
            print(f"[{jaw}] aucun scan")
            continue
        part, obj = scans[key][0]
        mesh = load_scan(obj)
        v = mesh.vertices
        lo, hi = v.min(axis=0), v.max(axis=0)
        print(f"[{jaw}] {part}/{obj.name} : {len(v)} sommets, {len(mesh.faces)} faces")
        print(f"    étendue (mm) x:{hi[0] - lo[0]:.1f}  y:{hi[1] - lo[1]:.1f}  z:{hi[2] - lo[2]:.1f}")
        print(f"    boîte englobante : min {np.round(lo, 1)}  max {np.round(hi, 1)}")
        print(f"    direction occlusale estimée : {np.round(open_side_direction(mesh), 2)}")
        lab = None
        if key in labels:
            lab, inst = load_labels(labels[key][0][1])
            ok = "OK" if len(lab) == len(v) else "INCOHÉRENT"
            teeth = sorted(int(t) for t in set(lab.tolist()) if t > 0)
            print(f"    labels : {len(lab)} valeurs ({ok}) ; {len(teeth)} dents FDI : {teeth}")
            if len(lab) != len(v):
                lab = None
        else:
            print("    labels : aucun")
        membership = [n for n, s in splits.items() if key in s] or ["aucun"]
        print(f"    découpage : {', '.join(membership)}")
        pts = None
        if key in kpts["train"]:
            d = json.loads(kpts["train"][key].read_text())
            c = Counter(o["class"] for o in d["objects"])
            pts = {cls: np.array([o["coord"] for o in d["objects"] if o["class"] == cls]) for cls in c}
            from scipy.spatial import cKDTree
            allp = np.vstack(list(pts.values()))
            dist, _ = cKDTree(v).query(allp)
            print(f"    landmarks (train) : {dict(c)}")
            print(f"    distance landmark -> sommet le plus proche : médiane {np.median(dist):.3f} mm, max {dist.max():.3f} mm")
        elif key in kpts["test"]:
            print("    landmarks : fichier de TEST présent (non lu : évaluation finale uniquement)")
        else:
            print("    landmarks : aucun")
        loaded[jaw] = (mesh, lab, pts)

    if len(loaded) == 2:
        m = occlusion_metrics(scans, labels, pid)
        if m:
            print("\nLes deux arcades sont-elles dans un repère commun (en occlusion) ?")
            print(f"    cosinus entre directions occlusales : {m['cos']:.2f}  (-1 = face à face, +1 = même sens)")
            print(f"    décalage haut/bas le long de la direction occlusale du bas : {m['offset']:.1f} mm")
            print(f"    distance entre centres de gravité des dents : {m['centroid_gap']:.1f} mm")
            print(f"    distance dents du haut -> dents du bas : min {m['d_min']:.2f} mm, médiane {m['d_med']:.2f} mm, "
                  f"{100 * m['near']:.1f} % des sommets à moins de 0,5 mm")
    return loaded


def show_patient(loaded, pid, screenshot: Path | None):
    """Affiche les arcades telles qu'elles sont dans les fichiers (aucun recalage)."""
    import pyvista as pv
    pl = pv.Plotter(off_screen=screenshot is not None, window_size=(1400, 900))
    colors = {"upper": "#e8d9b5", "lower": "#b5cfe8"}
    for jaw, (mesh, lab, pts) in loaded.items():
        faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64)
        poly = pv.PolyData(np.asarray(mesh.vertices), faces)
        if lab is not None:
            poly["dent"] = (lab > 0).astype(np.uint8)  # 0 = gencive, 1 = dent
            pl.add_mesh(poly, scalars="dent", cmap=["#d98c8c", colors[jaw]], show_scalar_bar=False,
                        smooth_shading=True, label=f"{jaw} (dents) / gencive en rose")
        else:
            pl.add_mesh(poly, color=colors[jaw], smooth_shading=True, label=jaw)
        if pts and "Cusp" in pts:
            pl.add_points(pts["Cusp"], color="red", point_size=10, render_points_as_spheres=True,
                          label=f"cuspides annotées ({jaw})")
    pl.add_axes()
    pl.show_bounds(grid=False, xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)")
    pl.add_legend(bcolor="white", size=(0.42, 0.16))
    pl.add_text(f"Patient {pid} - coordonnées brutes des fichiers, sans recalage", font_size=10)
    if screenshot is not None:
        pl.view_xz()
        pl.show(screenshot=str(screenshot))
        print(f"Capture enregistrée : {screenshot}")
    else:
        pl.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits"))
    ap.add_argument("--patient", help="identifiant patient à détailler (ex. 0140W3ND)")
    ap.add_argument("--show", action="store_true", help="vue 3D interactive du patient")
    ap.add_argument("--screenshot", type=Path, help="enregistre une image au lieu d'ouvrir la fenêtre")
    ap.add_argument("--occlusion-sample", type=int, default=0, metavar="N",
                    help="test d'occlusion chiffré sur N patients répartis dans le jeu de données")
    ap.add_argument("--no-inventory", action="store_true", help="saute l'inventaire complet")
    args = ap.parse_args()
    # Sortie en UTF-8 pour que les accents s'affichent correctement sous Windows.
    sys.stdout.reconfigure(encoding="utf-8")

    raw = args.raw.resolve()
    if not (raw / "teeth3ds").is_dir():
        sys.exit(f"Dossier introuvable : {raw / 'teeth3ds'}")
    if args.screenshot is not None:
        shot = args.screenshot.resolve()
        if raw in shot.parents:
            sys.exit("Refus : on n'écrit jamais dans data/raw.")
        shot.parent.mkdir(parents=True, exist_ok=True)

    scans, labels, kpts, unknown = build_index(raw)
    splits = read_splits(args.splits)
    if not args.no_inventory:
        report_inventory(scans, labels, kpts, unknown, splits)
    if args.occlusion_sample:
        report_occlusion_sample(scans, labels, args.occlusion_sample)
    if args.patient:
        loaded = report_patient(scans, labels, kpts, splits, args.patient)
        if loaded and (args.show or args.screenshot):
            show_patient(loaded, args.patient, args.screenshot.resolve() if args.screenshot else None)


if __name__ == "__main__":
    main()
