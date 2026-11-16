# OccluMap

Prototype de logiciel d'aide à la décision en prothèse dentaire, à partir de scans intra-oraux des arcades dentaires.

> **Prototype de recherche. Ce n'est pas un dispositif médical.**
> Aucune décision clinique ne doit dépendre de ce logiciel. Il n'a fait l'objet d'aucune évaluation clinique.

Projet étudiant de 5e année d'école d'ingénieurs, mené avec un partenaire clinique étudiant en chirurgie dentaire.

## Principe

L'outil compare la surface des dents à une surface de référence géométrique, la **sphère de Monson** (rayon classique d'environ 102 mm), pour indiquer quelles dents sont trop hautes ou trop basses par rapport à cette référence.

1. Chargement d'un scan d'arcade (maillage 3D).
2. Repérage des pointes cuspidiennes (clic manuel, ou annotations du jeu de données).
3. Ajustement robuste d'une sphère sur ces points (RANSAC) : les cuspides trop éloignées sont écartées comme valeurs aberrantes.
4. Distance signée de chaque point de la surface à la sphère.
5. À venir : carte « à réduire / à rehausser » et tableau par dent.

Le rayon de la sphère et le seuil de tolérance sont des paramètres réglables : aucune valeur n'est figée à ce stade.

## État d'avancement

| Élément | État |
|---|---|
| Vérification des données (inventaire, correspondance scans / annotations) | fait |
| Ajustement de sphère (moindres carrés, RANSAC, rayon libre ou imposé) | fait, vérifié sur données synthétiques uniquement |
| Visualiseur avec clic des cuspides | fait |
| Carte « réduire / rehausser » et tableau par dent | à faire |
| Détection automatique des cuspides | à faire |
| Évaluation avec des praticiens | à faire |

## Installation

Python 3.11, testé sous Windows.

```powershell
git clone https://github.com/AlexHAGT/OccluMap.git
cd OccluMap
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Données

**Ce dépôt ne contient aucune donnée.** Les scans ne sont pas redistribués ici et ne doivent pas l'être : le jeu de données Teeth3DS est sous licence CC BY-NC-ND 4.0 (usage non commercial, pas de redistribution, même modifiée).

Pour utiliser les scripts, il faut télécharger soi-même les données auprès de leurs auteurs :

- Teeth3DS+ et 3DTeethLand : <https://osf.io/xctdy/>
- Dépôt officiel (description, découpage entraînement / test) : `abenhamadou/3DTeethSeg_MICCAI_Challenges`

Organisation attendue, à la racine du projet :

```
data/
├── raw/
│   ├── teeth3ds/
│   │   ├── data_part_1/        (chaque archive dézippée telle quelle)
│   │   │   ├── upper/<ID>/<ID>_upper.obj, <ID>_upper.json
│   │   │   └── lower/<ID>/<ID>_lower.obj, <ID>_lower.json
│   │   └── data_part_2/ ... data_part_7/
│   └── 3dteethland/
│       ├── train/<upper|lower>/<ID>/<ID>_<upper|lower>__kpt.json
│       └── test/   (réservé à l'évaluation finale, jamais lu par les scripts d'ajustement)
└── splits/         (training_/testing_ upper/lower .txt du découpage officiel)
```

Les dossiers `data/`, `models/` et `outputs/` sont exclus du dépôt par le `.gitignore`.

## Utilisation

Toutes les commandes se lancent depuis la racine du projet.

**Vérifier les données**

```powershell
.\.venv\Scripts\python.exe scripts\check_teeth3ds.py
.\.venv\Scripts\python.exe scripts\check_teeth3ds.py --no-inventory --patient <ID> --show
```

**Afficher un scan, cliquer les cuspides, ajuster la sphère**

```powershell
.\.venv\Scripts\python.exe scripts\view_and_pick.py <ID>_lower --radius 102
```

- clic droit sur une cuspide : ajouter un point ;
- `z` : retirer le dernier point, `x` : tout effacer ;
- `espace` : ajuster la sphère, `q` : quitter.

Options : `--radius` (rayon imposé en mm ; absent = rayon libre), `--threshold` (écart toléré en mm), `--no-ransac`, `--use-annotations` (part des cuspides annotées d'un scan d'entraînement au lieu de cliquer).

**Essai de l'ajustement sur données synthétiques**

```powershell
.\.venv\Scripts\python.exe -m occlumap.sphere_fit
```

## Organisation du code

```
occlumap/           le logiciel
├── io.py           chargement des scans et des annotations
└── sphere_fit.py   ajustement de sphère (moindres carrés, RANSAC), distance signée
scripts/            outils en ligne de commande
├── check_teeth3ds.py
└── view_and_pick.py
```

## Limites connues

- L'ajustement n'a été vérifié que sur des données synthétiques ; aucun résultat d'évaluation sur données réelles n'est disponible.
- Dans Teeth3DS, les arcades du haut et du bas d'un même patient ne sont pas en occlusion : chaque arcade est traitée séparément.
- À rayon libre, le rayon estimé est très sensible au bruit, car les cuspides ne couvrent qu'une petite portion de la sphère.

## Références des données

- Teeth3DS+ : Ben-Hamadou et al., arXiv:2210.06094.
- 3DTeethLand : arXiv:2512.08323.
