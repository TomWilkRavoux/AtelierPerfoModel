# Détection de pneumonie sur radiographies thoraciques

Trois modèles de Deep Learning pour classer des radiographies pulmonaires, comparés sur le même jeu de test, avec une analyse détaillée de leurs métriques.

- **Modèle A** : un CNN conçu et entraîné de zéro (NORMAL / PNEUMONIA).
- **Modèle B** : un ResNet18 pré-entraîné sur ImageNet, en transfer learning (NORMAL / PNEUMONIA).
- **Modèle C** : le modèle A affiné pour distinguer 3 classes (NORMAL / BACTÉRIENNE / VIRALE).

Atelier de Deep Learning réalisé en juin 2026 dans le cadre du master IA de l'EPSI. Contrainte choisie : Deep Learning uniquement, sans ML classique.

## Résultats

Jeu de test : 624 radiographies (234 normales, 390 pneumonies).

| Métrique | Modèle A (CNN maison) | Modèle B (ResNet18) |
| --- | --- | --- |
| Accuracy | 0,729 | **0,848** |
| Précision (PNEUMONIA) | 0,699 | **0,811** |
| Rappel / sensibilité | **0,995** | 0,987 |
| Spécificité | 0,286 | **0,615** |
| F1-score | 0,821 | **0,890** |
| ROC-AUC | 0,919 | **0,952** |

**Modèle B retenu** pour la tâche binaire : sensibilité quasi identique (il rate aussi peu de pneumonies), mais spécificité deux fois meilleure, donc beaucoup moins de patients sains signalés à tort.

**Modèle C (3 classes)** : accuracy 0,678, F1 macro 0,647. Les pneumonies bactériennes sont bien reconnues (F1 0,874) ; les cas normaux sont sous-détectés et les pneumonies virales en partie confondues, une limite attendue quand on ne dispose que de la radiographie.

## Démarche

1. **Exploration** : déséquilibre d'environ 3 pneumonies pour 1 cas normal, images en niveaux de gris de tailles très variables, et un jeu de validation d'origine inutilisable (16 images).
2. **Préparation** : fusion de train et validation puis nouveau découpage stratifié 85/15 (4 447 images d'entraînement, 785 de validation), redimensionnement en 224 × 224, normalisation ImageNet, augmentation sur l'entraînement seulement (flip horizontal, rotation ±10°), déséquilibre compensé par un `WeightedRandomSampler`.
3. **Modèles** : un CNN de 4 blocs Conv-BN-ReLU-MaxPool (environ 0,5 M de paramètres), puis un ResNet18 au backbone gelé, puis le CNN affiné en 3 classes.
4. **Analyse** : pour chaque modèle, matrice de confusion, courbe ROC et explication de chaque métrique. L'accuracy seule est trompeuse ici : un modèle qui répondrait toujours « pneumonie » ferait déjà 62 %.

## Installation

Prérequis : Python 3.12 et [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Les données ne sont pas versionnées. Télécharger le jeu Kaggle [*Chest X-Ray Images (Pneumonia)*](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) et le décompresser dans `asset/` :

```
asset/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

## Lancer les notebooks

Les notebooks sont des fichiers [marimo](https://marimo.io) (`.py` versionnables). Ils s'entraînent sur GPU CUDA si disponible, sinon sur CPU. Les modèles entraînés sont enregistrés dans `models/` (non versionné).

```bash
uv run marimo edit Exploration.py    # exploration, préparation, Modèle A
uv run marimo edit ModelB.py         # Modèle B (ResNet18) et comparaison A / B
uv run marimo edit PneumoniaType.py  # Modèle C (3 classes)
```

Le Modèle C repart des poids du Modèle A : lancer `Exploration.py` avant `PneumoniaType.py`. Dans `ModelB.py`, passer `FREEZE_BACKBONE = False` pour un fine-tuning complet du ResNet18 (plus long).

## Structure

```
├── Exploration.py       # notebook : données, préparation, Modèle A
├── ModelB.py            # notebook : Modèle B (transfer learning)
├── PneumoniaType.py     # notebook : Modèle C (3 classes)
├── chest_dataset.py     # Dataset PyTorch commun aux notebooks
├── model_a_cnn.py       # architecture SimpleCNN (Modèles A et C)
├── PLAN.md              # plan de travail et choix, phase par phase
└── docs/                # documentation détaillée par modèle
```

## Documentation

- [Index de la documentation](docs/README.md)
- Modèle A : [construction](docs/modele_a_cnn.md) · [analyse des métriques](docs/analyse_metrics_modele_a.md) · [métriques et pistes d'amélioration](docs/metriques_et_pistes.md)
- Modèle B : [construction](docs/modele_b_resnet18.md) · [analyse des métriques](docs/analyse_metrics_modele_b.md)
- Modèle C : [construction](docs/modele_c_pneumonia_type.md) · [analyse des métriques](docs/analyse_metrics_modele_c.md)

## Limites

- Projet pédagogique : ces modèles ne sont pas un outil de diagnostic médical.
- Évaluation sur le seul jeu de test Kaggle (624 images) ; aucune validation sur des radiographies d'un autre hôpital.
- Le Modèle B est entraîné backbone gelé ; un fine-tuning complet pourrait encore améliorer la spécificité.

## Stack

Python 3.12 · PyTorch · torchvision (ResNet18) · scikit-learn (métriques) · Polars · matplotlib · marimo · uv
