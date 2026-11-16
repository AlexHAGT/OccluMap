"""Ajustement d'une sphère sur un petit nuage de points (pointes cuspidiennes).

Trois briques, de la plus simple à la plus robuste :

1. fit_sphere_lstsq          : moindres carrés, rayon libre (tous les points comptent).
2. fit_sphere_fixed_radius   : moindres carrés, rayon imposé (ex. Monson, 102 mm).
3. ransac_sphere             : ajustement robuste. Les points trop loin de la sphère
                               (dents trop hautes / trop basses) sont écartés comme
                               valeurs aberrantes, puis la sphère est réajustée sur
                               les points conservés.

Convention : toutes les longueurs sont en millimètres.
Distance signée d'un point à la sphère : ||p - centre|| - rayon
    négatif = à l'intérieur de la sphère, positif = à l'extérieur.
L'interprétation « à réduire / à rehausser » dépend de l'arcade et n'est PAS faite ici.

Point d'attention : les cuspides d'une arcade ne couvrent qu'une petite calotte
d'une sphère d'environ 100 mm de rayon. Avec un rayon libre, le problème est mal
conditionné : un bruit de quelques dixièmes de mm déplace le rayon de plusieurs mm.
C'est pour cela que le rayon peut être imposé.

Validation : `python -m occlumap.sphere_fit` lance un essai sur données SYNTHÉTIQUES.
Ce module n'a pas été évalué sur des données cliniques.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SphereFit:
    center: np.ndarray            # (3,) mm
    radius: float                 # mm
    residuals: np.ndarray         # (n,) distance signée de chaque point, mm
    inliers: np.ndarray           # (n,) booléens : point conservé pour l'ajustement
    method: str = ""
    info: dict = field(default_factory=dict)

    @property
    def rms_inliers(self) -> float:
        """Écart quadratique moyen des points conservés (mm)."""
        return float(np.sqrt(np.mean(self.residuals[self.inliers] ** 2)))

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def signed_distance(points, center, radius) -> np.ndarray:
    """||p - centre|| - rayon, en mm (négatif = à l'intérieur de la sphère)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return np.linalg.norm(points - np.asarray(center, dtype=float), axis=1) - float(radius)


def _check_points(points, n_min) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points doit être un tableau (n, 3)")
    if len(points) < n_min:
        raise ValueError(f"il faut au moins {n_min} points, {len(points)} fournis")
    return points


# --------------------------------------------------------------------------- #
# 1. Moindres carrés, rayon libre
# --------------------------------------------------------------------------- #
def _algebraic_fit(points):
    """Ajustement « algébrique » : ||p||² = 2 p.c + (r² - ||c||²), linéaire en (c, k).

    Rapide et sans point de départ, mais légèrement biaisé quand les points ne
    couvrent qu'une petite calotte. Sert d'initialisation à l'ajustement géométrique.
    Les points sont centrés avant le calcul pour la stabilité numérique.
    """
    origin = points.mean(axis=0)
    q = points - origin
    A = np.hstack([2.0 * q, np.ones((len(q), 1))])
    b = (q ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c, k = sol[:3], sol[3]
    r2 = k + c @ c
    if not np.isfinite(r2) or r2 <= 0:
        raise np.linalg.LinAlgError("points dégénérés (coplanaires ou alignés)")
    return c + origin, float(np.sqrt(r2))


def fit_sphere_lstsq(points, refine: bool = True) -> SphereFit:
    """Sphère au sens des moindres carrés, rayon libre, sur TOUS les points.

    refine=True : après l'ajustement algébrique, on minimise la vraie distance
    géométrique sum (||p - c|| - r)² (problème non linéaire, résolu par scipy).
    """
    points = _check_points(points, 4)
    center, radius = _algebraic_fit(points)
    if refine and len(points) > 4:
        def fun(x):
            return np.linalg.norm(points - x[:3], axis=1) - x[3]
        res = least_squares(fun, np.append(center, radius))
        if res.success and res.x[3] > 0:
            center, radius = res.x[:3], float(res.x[3])
    return SphereFit(center, radius, signed_distance(points, center, radius),
                     np.ones(len(points), dtype=bool), method="moindres carrés, rayon libre")


# --------------------------------------------------------------------------- #
# 2. Moindres carrés, rayon imposé
# --------------------------------------------------------------------------- #
def _plane_normal(points):
    """Normale du plan moyen des points (direction de plus faible variance, par SVD)."""
    q = points - points.mean(axis=0)
    return np.linalg.svd(q, full_matrices=False)[2][-1]


def _resolve_side(points, side):
    """Vecteur unitaire indiquant de quel côté du nuage se trouve le centre.

    Avec un rayon imposé, deux sphères symétriques passent par une calotte donnée
    (centre « au-dessus » ou « en dessous »). `side` lève l'ambiguïté. S'il n'est
    pas fourni, on prend le côté trouvé par l'ajustement à rayon libre.
    """
    n = _plane_normal(points)
    if side is None:
        free_center, _ = _algebraic_fit(points)
        side = free_center - points.mean(axis=0)
    side = np.asarray(side, dtype=float)
    if abs(n @ side) < 1e-9:
        raise ValueError("`side` est parallèle au plan des points : côté indéterminé")
    return n if n @ side > 0 else -n


def fit_sphere_fixed_radius(points, radius: float, side=None) -> SphereFit:
    """Sphère de rayon imposé : seul le centre est ajusté (moindres carrés géométriques).

    side : vecteur (3,) pointant grossièrement des points vers le centre attendu.
    """
    points = _check_points(points, 3)
    n = _resolve_side(points, side)
    start = points.mean(axis=0) + radius * n

    def fun(c):
        return np.linalg.norm(points - c, axis=1) - radius
    res = least_squares(fun, start)
    center = res.x
    return SphereFit(center, float(radius), signed_distance(points, center, radius),
                     np.ones(len(points), dtype=bool), method="moindres carrés, rayon imposé",
                     info={"side": n})


def _sphere_through_3_points(a, b, c, radius, side_unit):
    """Centre de la sphère de rayon donné passant par 3 points, du côté `side_unit`.

    Les 3 points définissent un cercle (centre cc, rayon rho). Le centre de la
    sphère est sur l'axe de ce cercle, à la hauteur sqrt(r² - rho²).
    """
    u, v = b - a, c - a
    w = np.cross(u, v)
    w2 = w @ w
    if w2 < 1e-12:
        return None  # points alignés
    cc = a + ((u @ u) * np.cross(v, w) + (v @ v) * np.cross(w, u)) / (2.0 * w2)
    h2 = radius ** 2 - (cc - a) @ (cc - a)
    if h2 <= 0:
        return None  # cercle plus grand que la sphère
    n = w / np.sqrt(w2)
    if n @ side_unit < 0:
        n = -n
    return cc + np.sqrt(h2) * n


# --------------------------------------------------------------------------- #
# 3. RANSAC
# --------------------------------------------------------------------------- #
def ransac_sphere(points, threshold: float = 0.5, radius: float | None = None, side=None,
                  n_iter: int = 2000, radius_bounds=(30.0, 300.0), seed: int | None = 0) -> SphereFit:
    """Ajustement robuste d'une sphère (RANSAC), rayon libre ou imposé.

    Principe : on tire au hasard un échantillon minimal de points (4 si le rayon
    est libre, 3 s'il est imposé), on calcule la sphère qui passe par eux, et on
    compte les points situés à moins de `threshold` mm de cette sphère. On garde
    la sphère qui rassemble le plus de points, puis on la réajuste par moindres
    carrés sur ces points (« inliers »). Les autres sont les valeurs aberrantes.

    threshold      : écart toléré (mm) pour qu'un point soit conservé. Paramètre
                     à fixer avec le partenaire clinique ; 0,5 mm est un défaut provisoire.
    radius         : None = rayon libre ; sinon rayon imposé (mm).
    side           : voir fit_sphere_fixed_radius (utile seulement si radius est donné).
    radius_bounds  : en rayon libre, les sphères candidates hors de cet intervalle
                     sont rejetées (une calotte presque plane donne des rayons absurdes).
    seed           : graine du tirage aléatoire, pour un résultat reproductible.
    """
    fixed = radius is not None
    n_sample = 3 if fixed else 4
    points = _check_points(points, n_sample + 1)
    rng = np.random.default_rng(seed)
    side_unit = _resolve_side(points, side) if fixed else None

    best_mask, best_score = None, (-1, np.inf)
    n_valid = 0
    for _ in range(n_iter):
        sample = points[rng.choice(len(points), n_sample, replace=False)]
        if fixed:
            center = _sphere_through_3_points(*sample, radius, side_unit)
            if center is None:
                continue
            r = radius
        else:
            try:
                center, r = _algebraic_fit(sample)
            except np.linalg.LinAlgError:
                continue
            if not (radius_bounds[0] <= r <= radius_bounds[1]):
                continue
        n_valid += 1
        res = np.abs(signed_distance(points, center, r))
        mask = res < threshold
        score = (int(mask.sum()), float(res[mask].sum()))
        # Meilleur = plus d'inliers ; à égalité, inliers plus proches de la sphère.
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_mask, best_score = mask, score

    if best_mask is None or best_mask.sum() < n_sample + 1:
        raise RuntimeError("RANSAC : aucun consensus trouvé (trop peu de points, seuil trop strict "
                           "ou bornes de rayon inadaptées)")

    # Réajustement sur les inliers, puis reclassement des points ; répété jusqu'à stabilité.
    mask = best_mask
    for _ in range(10):
        fit = (fit_sphere_fixed_radius(points[mask], radius, side_unit) if fixed
               else fit_sphere_lstsq(points[mask]))
        new_mask = np.abs(signed_distance(points, fit.center, fit.radius)) < threshold
        if new_mask.sum() < n_sample + 1 or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    return SphereFit(fit.center, fit.radius, signed_distance(points, fit.center, fit.radius), mask,
                     method="RANSAC, rayon imposé" if fixed else "RANSAC, rayon libre",
                     info={"threshold": threshold, "n_iter": n_iter, "n_valid_models": n_valid})


# --------------------------------------------------------------------------- #
# Essai sur données synthétiques
# --------------------------------------------------------------------------- #
def synthetic_cusps(rng, radius=100.0, n=28, noise=0.1, n_outliers=5, outlier_range=(1.5, 3.0)):
    """Fausses cuspides : points sur une calotte en forme d'arcade, bruit + dents aberrantes.

    Renvoie (points, vrai_centre, masque_des_points_aberrants).
    """
    center = np.array([3.0, -5.0, radius - 90.0])
    t = np.linspace(-np.pi / 2, np.pi / 2, n)          # arcade en U vue de dessus
    x = 25.0 * np.sin(t) + rng.normal(0, 1.5, n)
    y = 40.0 * np.cos(t) + rng.normal(0, 1.5, n)
    z = center[2] - np.sqrt(radius ** 2 - (x - center[0]) ** 2 - (y - center[1]) ** 2)
    pts = np.column_stack([x, y, z])
    radial = (pts - center) / np.linalg.norm(pts - center, axis=1, keepdims=True)
    pts = pts + radial * rng.normal(0, noise, n)[:, None]
    out = np.zeros(n, dtype=bool)
    out[rng.choice(n, n_outliers, replace=False)] = True
    shift = rng.uniform(*outlier_range, n) * rng.choice([-1.0, 1.0], n)
    pts[out] += radial[out] * shift[out, None]
    return pts, center, out


def _demo(n_trials=100, radius=100.0):
    rng = np.random.default_rng(0)
    side = np.array([0.0, 0.0, 1.0])
    rows = {"moindres carrés, rayon libre (tous les points)": [],
            "RANSAC, rayon libre": [],
            "RANSAC, rayon imposé": []}
    detect = {"RANSAC, rayon libre": [], "RANSAC, rayon imposé": []}
    for i in range(n_trials):
        pts, c0, out = synthetic_cusps(rng, radius=radius)
        fits = {
            "moindres carrés, rayon libre (tous les points)": fit_sphere_lstsq(pts),
            "RANSAC, rayon libre": ransac_sphere(pts, radius=None, n_iter=1000, seed=i),
            "RANSAC, rayon imposé": ransac_sphere(pts, radius=radius, side=side, n_iter=1000, seed=i),
        }
        for name, f in fits.items():
            # Erreur utile en pratique : écart entre la sphère trouvée et la vraie,
            # mesuré sur les vrais points non aberrants (mm).
            true_pts = pts[~out]
            surf = np.abs(signed_distance(true_pts, f.center, f.radius)
                          - signed_distance(true_pts, c0, radius)).max()
            rows[name].append((abs(f.radius - radius), surf))
            if name in detect:
                detect[name].append(np.array_equal(~f.inliers, out))
    print(f"Données SYNTHÉTIQUES : {n_trials} essais, 28 points sur une calotte de rayon {radius:.0f} mm,")
    print("bruit radial 0,1 mm (écart-type), 5 points aberrants décalés de 1,5 à 3 mm, seuil 0,5 mm.\n")
    print(f"{'méthode':<48}{'err. rayon méd.':>16}{'err. rayon p95':>16}{'écart surf. méd.':>18}{'écart surf. p95':>17}")
    for name, vals in rows.items():
        v = np.array(vals)
        print(f"{name:<48}{np.median(v[:, 0]):>13.2f} mm{np.percentile(v[:, 0], 95):>13.2f} mm"
              f"{np.median(v[:, 1]):>15.3f} mm{np.percentile(v[:, 1], 95):>14.3f} mm")
    print("\n« écart surf. » : plus grand écart entre la sphère trouvée et la vraie sphère,")
    print("mesuré au niveau des points non aberrants.")
    for name, ok in detect.items():
        print(f"{name} : points aberrants retrouvés exactement dans {sum(ok)} essais sur {n_trials}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    _demo()
