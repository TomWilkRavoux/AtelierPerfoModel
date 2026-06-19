# Documentation — Détection de pneumonie (Deep Learning)

Index de la documentation du projet : conception, entraînement et analyse de
plusieurs modèles de Deep Learning pour classifier des radiographies thoraciques.

---

## Les modèles

| Modèle | Tâche | Approche | Notebook |
|---|---|---|---|
| **A** | NORMAL / PNEUMONIA | CNN maison (*from scratch*) | [`../Exploration.py`](../Exploration.py) |
| **B** | NORMAL / PNEUMONIA | Transfer learning (ResNet18 pré-entraîné) | [`../ModelB.py`](../ModelB.py) |
| **C** | NORMAL / BACTÉRIENNE / VIRALE | Fine-tuning du Modèle A en 3 classes | [`../PneumoniaType.py`](../PneumoniaType.py) |

---

## Documentation par modèle

### Modèle A — CNN maison
- [Construction & entraînement](modele_a_cnn.md) — architecture, pipeline, boucle d'entraînement, sauvegarde.
- [Analyse des métriques](analyse_metrics_modele_a.md) — pourquoi acc 0,73 / sensibilité 0,99 / spécificité 0,29.
- [Métriques & pistes d'amélioration](metriques_et_pistes.md) — explication des métriques + leviers (seuil, filtres, transfer learning…).

### Modèle B — ResNet18 (transfer learning)
- [Construction & entraînement](modele_b_resnet18.md) — transfer learning, backbone gelé, tête remplacée.
- [Analyse des métriques](analyse_metrics_modele_b.md) — pourquoi B domine A (acc 0,85, spécificité 0,62, AUC 0,95).

### Modèle C — 3 classes (sain / bactérien / viral)
- [Construction & entraînement](modele_c_pneumonia_type.md) — rechargement du Modèle A, bascule de tête 2→3, fine-tuning.
- [Analyse des métriques](analyse_metrics_modele_c.md) — pourquoi acc 0,68 / F1 macro 0,65, et la confusion NORMAL ↔ VIRALE.

---

## Récapitulatif des performances (set de test, 624 images)

### Tâche binaire (NORMAL / PNEUMONIA)
| Métrique | Modèle A | Modèle B |
|---|---|---|
| Accuracy | 0,729 | **0,848** |
| Précision (PNEUMONIA) | 0,699 | **0,811** |
| Rappel / Sensibilité | **0,995** | 0,987 |
| Spécificité | 0,286 | **0,615** |
| F1-score | 0,821 | **0,890** |
| ROC-AUC | 0,919 | **0,952** |

→ **Modèle B retenu** pour le binaire : meilleur sur quasi toutes les métriques,
sensibilité quasi égale et spécificité nettement supérieure.

### Tâche 3 classes (Modèle C)
| | Précision | Rappel | F1 |
|---|---|---|---|
| NORMAL | 0,973 | 0,312 | 0,473 |
| BACTÉRIENNE | 0,837 | 0,913 | 0,874 |
| VIRALE | 0,453 | 0,872 | 0,596 |

Accuracy 0,678 · **F1 macro 0,647**. Bactérien bien reconnu ; NORMAL sous-détecté
et viral/bactérien partiellement confondus (limite attendue de la radio seule).

---

## Fichiers de code associés
- [`../chest_dataset.py`](../chest_dataset.py) — `Dataset` PyTorch (module importable, picklable par les workers).
- [`../model_a_cnn.py`](../model_a_cnn.py) — architecture `SimpleCNN` (réutilisée par les Modèles A et C).
- Modèles entraînés (non versionnés) : `models/model_a_cnn.pt`, `models/model_b_resnet18.pt`, `models/model_c_pneumonia_type.pt`.
