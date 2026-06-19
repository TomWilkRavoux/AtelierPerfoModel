# Modèle A — CNN maison : construction & entraînement

> Documentation du **Modèle A** (baseline Deep Learning) pour la détection de
> pneumonie sur radiographies thoraciques. Le code correspondant se trouve dans
> le notebook marimo [`Exploration.py`](../Exploration.py).

---

## 1. Vue fonctionnelle

### 1.1 Objectif
Classer automatiquement une **radiographie pulmonaire** dans l'une des deux
catégories :

- `NORMAL` — poumon sain ;
- `PNEUMONIA` — présence d'une pneumonie.

Le Modèle A est un **réseau de neurones convolutif (CNN) entraîné de zéro**
(*from scratch*). Il sert de **point de référence (baseline)** : on mesurera
plus tard l'apport d'une approche par *transfer learning* (Modèle B) en la
comparant à ce modèle.

### 1.2 Principe général
Un CNN apprend tout seul, à partir des images annotées, les **motifs visuels**
qui distinguent un poumon sain d'un poumon atteint (opacités, zones denses…).
Le réseau procède en deux temps :

1. **Extraction de caractéristiques** : des couches de convolution transforment
   l'image en cartes de caractéristiques de plus en plus abstraites (bords →
   textures → structures).
2. **Décision** : un classifieur dense prend ces caractéristiques et produit
   un score pour chacune des deux classes.

### 1.3 Entrée / sortie
| | Description |
|---|---|
| **Entrée** | Une image RGB redimensionnée en `224 × 224` pixels, normalisée. |
| **Sortie** | 2 *logits* (scores bruts) — un par classe. La classe prédite est celle au score le plus élevé. |

---

## 2. Données utilisées

Dataset Kaggle *Chest X-Ray Images (Pneumonia)*, situé dans `asset/chest_xray/`.

| Split (après re-split) | NORMAL | PNEUMONIA | Total |
|------------------------|--------|-----------|-------|
| train                  | —      | —         | 4447  |
| val                    | —      | —         | 785   |
| test                   | 234    | 390       | 624   |

Deux particularités traitées en amont (Phase 2) :

- **Déséquilibre des classes** (~3× plus de pneumonies que de cas normaux) ;
- **Validation d'origine inutilisable** (16 images seulement) → reconstruite par
  un **re-split stratifié 85/15** fusionnant les `train`+`val` d'origine.

> Voir le pipeline de préparation (transforms, augmentation, gestion du
> déséquilibre, `DataLoader`) dans `Exploration.py`, Phase 2.

---

## 3. Vue technique — Architecture du modèle

Classe `SimpleCNN` (`torch.nn.Module`). Entrée `(B, 3, 224, 224)` → sortie
`(B, 2)`.

### 3.1 Bloc de convolution réutilisable
Chaque bloc enchaîne :

```
Conv2d(3×3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2)
```

- **Conv2d 3×3, padding=1** : détecte des motifs locaux sans changer la
  résolution spatiale (padding conserve la taille).
- **BatchNorm2d** : stabilise et accélère l'apprentissage (normalise les
  activations par batch).
- **ReLU** : non-linéarité.
- **MaxPool2d(2)** : divise la résolution par 2 → réduit le coût et agrandit le
  champ réceptif.

### 3.2 Empilement
| Étage | Opération | Canaux | Taille spatiale |
|-------|-----------|--------|-----------------|
| Entrée | — | 3 | 224 × 224 |
| Bloc 1 | conv block | 32 | 112 × 112 |
| Bloc 2 | conv block | 64 | 56 × 56 |
| Bloc 3 | conv block | 128 | 28 × 28 |
| Bloc 4 | conv block | 128 | 14 × 14 |
| Pool | `AdaptiveAvgPool2d(4×4)` | 128 | 4 × 4 |
| Classifieur | `Flatten → Dropout(0.4) → Linear(2048→128) → ReLU → Dropout(0.3) → Linear(128→2)` | — | 2 logits |

- **`AdaptiveAvgPool2d((4, 4))`** : force une sortie spatiale fixe (4×4)
  indépendamment des petites variations, ce qui fige la taille d'entrée du
  classifieur (`128 × 4 × 4 = 2048`).
- **Dropout (0.4 puis 0.3)** : régularisation contre le sur-apprentissage.

### 3.3 Caractéristiques
- **~0.50 M paramètres** (≈ 504 066) — modèle volontairement léger.
- Sortie validée par une passe avant sur batch factice : `(2, 2)`.

---

## 4. Vue technique — Entraînement

### 4.1 Configuration
| Élément | Valeur | Rôle |
|---------|--------|------|
| Perte | `CrossEntropyLoss` | Adaptée à la classification multi-classes (ici 2). |
| Optimiseur | `Adam` (lr = `1e-3`) | Descente de gradient adaptative. |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2) | Baisse le LR quand la val stagne. |
| Epochs max | `30` | Plafond ; l'early stopping coupe souvent avant. |
| Early stopping | patience `5` | Arrêt si la val loss ne s'améliore plus pendant 5 epochs. |
| Batch size | `32` | — |
| `num_workers` | `8` | Lecture des images en parallèle (8 cœurs CPU). |
| Device | CPU | CUDA indisponible sur la machine. |

> Le déséquilibre des classes est géré **en amont** par un
> `WeightedRandomSampler` (les deux classes sont tirées de façon ~équilibrée),
> donc la perte n'est volontairement **pas** pondérée pour éviter une double
> correction. Un `class_weights_tensor` reste disponible comme alternative.

### 4.2 Déroulé d'une epoch — `run_epoch(model, loader, criterion, optimizer=None)`
Fonction unique pour l'entraînement **et** l'évaluation :

- si `optimizer` est fourni → mode **train** : `model.train()`, gradients
  activés, `zero_grad → forward → loss → backward → step` ;
- sinon → mode **éval** : `model.eval()`, gradients désactivés
  (`torch.set_grad_enabled(False)`), aucune mise à jour des poids.

Retourne `(loss moyenne, accuracy)` sur l'ensemble parcouru.

### 4.3 Boucle complète — `train_model(...)`
Pour chaque epoch :

1. une passe d'entraînement (`run_epoch` avec optimiseur) ;
2. une passe de validation (`run_epoch` sans optimiseur) ;
3. `scheduler.step(val_loss)` ;
4. enregistrement des métriques dans `history` ;
5. si la **val loss** s'améliore → sauvegarde des poids (`best_state`),
   sinon incrément du compteur d'early stopping ;
6. arrêt anticipé si patience dépassée.

À la fin, retourne un dictionnaire :

```python
{
    "history": {"train_loss", "train_acc", "val_loss", "val_acc"},  # listes
    "best_state": <state_dict des meilleurs poids>,
    "best_val_loss": <float>,
}
```

Le modèle recharge ensuite ses **meilleurs poids** :
`model_a.load_state_dict(training_result["best_state"])`.

### 4.4 Suivi
Deux courbes sont tracées en fin d'entraînement : **loss** (train vs val) et
**accuracy** (train vs val), pour diagnostiquer le sur/sous-apprentissage.

---

## 5. Sauvegarde du modèle

Le modèle entraîné est sérialisé via `torch.save` dans `models/model_a_cnn.pt`
(dossier ignoré par git). Le fichier contient :

```python
{
    "model_state": <best_state>,      # poids du modèle
    "history": <history>,             # courbes d'apprentissage
    "classes": ["NORMAL", "PNEUMONIA"],
    "img_size": 224,
    "arch": "SimpleCNN",
}
```

### Rechargement
```python
import torch
checkpoint = torch.load("models/model_a_cnn.pt", weights_only=False)

model = SimpleCNN(num_classes=2)
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

---

## 6. Pour aller plus loin
- **Phase 5** : évaluation sur le set de test (matrice de confusion, rappel /
  sensibilité, précision, F1, ROC-AUC).
- **Modèle B** : *transfer learning* (ResNet18 / DenseNet121) dans un notebook
  séparé, puis comparaison des deux approches.
