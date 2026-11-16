"""Affiche un scan, permet de cliquer les pointes cuspidiennes et ajuste une sphère.

Usage (depuis la racine du projet, avec le venv) :
    python scripts/view_and_pick.py 0140W3ND_lower                 # clic manuel des cuspides
    python scripts/view_and_pick.py 0140W3ND_lower --radius 102    # rayon imposé (Monson)
    python scripts/view_and_pick.py 0140W3ND_lower --use-annotations   # cuspides annotées (train)
    python scripts/view_and_pick.py chemin/vers/scan.obj --jaw lower

Dans la fenêtre :
    clic droit sur une cuspide : ajoute un point
    z : retire le dernier point          x : efface tous les points
    espace : ajuste la sphère            q : quitter

Couleurs après ajustement : distance signée de chaque sommet à la sphère (mm),
négatif = à l'intérieur de la sphère, positif = à l'extérieur. Cette vue ne dit pas
encore « à réduire / à rehausser » (étape suivante du projet).

Prototype de recherche : ce n'est pas un dispositif médical.
Lecture seule sur data/raw ; la seule écriture possible est --screenshot (refusée dans data/raw).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # pour importer le paquet occlumap sans l'installer

from occlumap import io  # noqa: E402
from occlumap.sphere_fit import fit_sphere_fixed_radius, fit_sphere_lstsq, ransac_sphere, signed_distance  # noqa: E402

CLIM = 2.0          # bornes de l'échelle de couleurs, en mm
CAP_EXTENT = 45.0   # on ne dessine la sphère qu'à moins de 45 mm du centre des cuspides


def center_side(jaw, occlusal):
    """De quel côté de l'arcade se trouve le centre de la sphère de Monson.

    Le centre est au-dessus du plan occlusal (côté crâne). Pour l'arcade du bas,
    c'est le côté des couronnes ; pour celle du haut, le côté des racines.
    Renvoie None si l'arcade est inconnue (le côté est alors déduit des points).
    """
    if jaw == "lower":
        return occlusal
    if jaw == "upper":
        return -occlusal
    return None


def fit_points(points, args, side):
    """Choisit la méthode selon le nombre de points et les options."""
    n_min = 3 if args.radius else 4
    if len(points) < n_min:
        raise ValueError(f"il faut au moins {n_min} points ({len(points)} placés)")
    # RANSAC n'a de sens qu'avec nettement plus de points que l'échantillon minimal.
    if args.no_ransac or len(points) < n_min + 3:
        if args.radius:
            return fit_sphere_fixed_radius(points, args.radius, side)
        return fit_sphere_lstsq(points)
    return ransac_sphere(points, threshold=args.threshold, radius=args.radius, side=side)


def describe(fit, points, occlusal):
    c = fit.center
    print(f"\nAjustement : {fit.method}")
    print(f"    points : {len(points)} ; conservés : {fit.n_inliers} ; écartés : {len(points) - fit.n_inliers}")
    print(f"    rayon  : {fit.radius:.1f} mm")
    print(f"    centre : [{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}] mm")
    side = "côté couronnes" if (c - points.mean(axis=0)) @ occlusal > 0 else "côté racines"
    print(f"    le centre est du {side}")
    print(f"    écart quadratique moyen des points conservés : {fit.rms_inliers:.2f} mm")
    for i in np.flatnonzero(~fit.inliers):
        print(f"    point {i + 1} écarté : distance signée {fit.residuals[i]:+.2f} mm")


class Viewer:
    def __init__(self, mesh, args, title):
        import pyvista as pv
        self.pv = pv
        self.args = args
        self.occlusal = io.occlusal_direction(mesh)
        self.side = center_side(args.jaw, self.occlusal)
        faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64)
        self.poly = pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)
        self.points: list[np.ndarray] = []
        self.fit = None
        self.pl = pv.Plotter(off_screen=args.screenshot is not None, window_size=(1400, 900))
        self.pl.add_mesh(self.poly, color="#e8dcc0", smooth_shading=True, name="scan")
        self.pl.add_axes()
        self.pl.add_text(title, font_size=10, name="title")
        self.pl.add_text("Prototype de recherche - pas un dispositif médical",
                         position="lower_right", font_size=8, name="disclaimer")
        self.status("clic droit : ajouter une cuspide - z : annuler - x : tout effacer - espace : ajuster")

    def status(self, text):
        self.pl.add_text(text, position="lower_left", font_size=9, name="status")

    def draw_points(self):
        if not self.points:
            self.pl.remove_actor("cusps")
            return
        pts = np.array(self.points)
        cloud = self.pv.PolyData(pts)
        if self.fit is not None and len(self.fit.inliers) == len(pts):
            cloud["conservé"] = self.fit.inliers.astype(np.uint8)
            self.pl.add_mesh(cloud, scalars="conservé", cmap=["red", "limegreen"], clim=[0, 1],
                             show_scalar_bar=False, point_size=12, render_points_as_spheres=True,
                             name="cusps", pickable=False, reset_camera=False)
        else:
            self.pl.add_mesh(cloud, color="black", point_size=12, render_points_as_spheres=True,
                             name="cusps", pickable=False, reset_camera=False)

    def on_pick(self, point):
        if point is None:
            return
        self.points.append(np.asarray(point, dtype=float))
        self.fit = None
        self.draw_points()
        self.status(f"{len(self.points)} point(s) - espace : ajuster")

    def undo(self):
        if self.points:
            self.points.pop()
            self.fit = None
            self.draw_points()
            self.status(f"{len(self.points)} point(s)")

    def clear(self):
        self.points.clear()
        self.fit = None
        self.draw_points()
        self.pl.remove_actor("sphere")
        self.pl.add_mesh(self.poly, color="#e8dcc0", smooth_shading=True, name="scan", reset_camera=False)
        self.status("points effacés")

    def do_fit(self):
        pts = np.array(self.points).reshape(-1, 3)
        try:
            self.fit = fit_points(pts, self.args, self.side)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"Ajustement impossible : {e}")
            self.status(f"Ajustement impossible : {e}")
            return
        fit = self.fit
        describe(fit, pts, self.occlusal)
        # Carte de distance signée sur tout le maillage (palette violet/orange,
        # volontairement différente du rouge/bleu réservé à « réduire / rehausser »).
        self.poly["distance"] = signed_distance(self.poly.points, fit.center, fit.radius)
        self.pl.add_mesh(self.poly, scalars="distance", cmap="PuOr", clim=[-CLIM, CLIM],
                         smooth_shading=True, name="scan", reset_camera=False,
                         scalar_bar_args={"title": "distance signée à la sphère (mm)"})
        # Portion de sphère proche de l'arcade (la sphère entière ferait ~200 mm de large).
        sphere = self.pv.Sphere(radius=fit.radius, center=fit.center, theta_resolution=240, phi_resolution=240)
        near = np.linalg.norm(sphere.points - pts.mean(axis=0), axis=1) < CAP_EXTENT
        cap = sphere.extract_points(near, adjacent_cells=False).extract_surface(algorithm="dataset_surface")
        self.pl.add_mesh(cap, color="gray", opacity=0.25, name="sphere", pickable=False, reset_camera=False)
        self.draw_points()
        self.status(f"{fit.method} - rayon {fit.radius:.1f} mm - {fit.n_inliers}/{len(pts)} points conservés "
                    f"(vert) - écart RMS {fit.rms_inliers:.2f} mm")

    def run(self):
        if self.args.screenshot is not None:
            self.pl.view_isometric()
            self.pl.show(screenshot=str(self.args.screenshot))
            print(f"Capture enregistrée : {self.args.screenshot}")
            return
        self.pl.enable_surface_point_picking(callback=self.on_pick, show_message=False,
                                             show_point=False, left_clicking=False)
        self.pl.add_key_event("z", self.undo)
        self.pl.add_key_event("x", self.clear)
        self.pl.add_key_event("space", self.do_fit)
        self.pl.show()


def main():
    ap = argparse.ArgumentParser(description="Affichage d'un scan, clic des cuspides, ajustement de sphère.")
    ap.add_argument("scan", help="nom de scan (ex. 0140W3ND_lower) ou chemin d'un fichier .obj")
    ap.add_argument("--jaw", choices=["upper", "lower"], help="arcade (déduite du nom si possible)")
    ap.add_argument("--radius", type=float, help="rayon imposé en mm (ex. 102) ; absent = rayon libre")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="écart toléré en mm pour conserver un point (défaut provisoire : 0,5)")
    ap.add_argument("--no-ransac", action="store_true", help="moindres carrés simples, sans rejet de points")
    ap.add_argument("--use-annotations", action="store_true",
                    help="part des cuspides annotées (landmarks d'entraînement) au lieu de cliquer")
    ap.add_argument("--screenshot", type=Path, help="enregistre une image au lieu d'ouvrir la fenêtre "
                                                     "(avec --use-annotations)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    path = Path(args.scan)
    if path.suffix.lower() != ".obj":
        path = io.find_scan(args.scan)
    name = path.stem
    if args.jaw is None and name.rsplit("_", 1)[-1] in ("upper", "lower"):
        args.jaw = name.rsplit("_", 1)[-1]
    if args.screenshot is not None:
        args.screenshot = args.screenshot.resolve()
        if (ROOT / "data" / "raw").resolve() in args.screenshot.parents:
            sys.exit("Refus : on n'écrit jamais dans data/raw.")
        if not args.use_annotations:
            sys.exit("--screenshot n'a de sens qu'avec --use-annotations (pas de clic possible).")
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)

    mesh = io.load_scan(path)
    print(f"{name} : {len(mesh.vertices)} sommets, arcade : {args.jaw or 'inconnue'}")
    viewer = Viewer(mesh, args, f"{name} - rayon {'imposé ' + str(args.radius) + ' mm' if args.radius else 'libre'}")

    if args.use_annotations:
        kpt = io.find_train_landmarks(name)
        if kpt is None:
            sys.exit(f"Aucun landmark d'entraînement pour {name} (les landmarks de test ne sont jamais utilisés ici).")
        viewer.points = list(io.load_landmarks(kpt, "Cusp"))
        print(f"{len(viewer.points)} cuspides annotées chargées")
        viewer.do_fit()
    viewer.run()


if __name__ == "__main__":
    main()
