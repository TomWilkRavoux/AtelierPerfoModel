# Modèle B — ResNet18 (transfer learning) : construction & entraînement

> Documentation du **Modèle B** (transfer learning) pour la détection de
> pneumonie sur radiographies thoraciques. Le code se trouve dans le notebook
> marimo [`ModelB.py`](../ModelB.py).
>
> Voir aussi : [analyse des métriques du Modèle B](analyse_metrics_modele_b.md).

---

## 1. Vue fonctionnelle

### 1.1 Objectif
Même tâche que le Modèle A : classer une **radiographie pulmonaire** en
`NORMAL` ou `PNEUMONIA`. La différence est l'**approche** : au lieu d'entraîner un
réseau de zéro, on **réutilise un réseau déjà entraîné** sur des millions
d'images.

### 1.2 Principe : le transfer learning
Un réseau **pré-entraîné sur ImageNet** (1,2 M d'images, 1000 catégories) a déjà
appris à détecter des **motifs visuels génériques** (bords, textures, formes).
Ces compétences se **transfèrent** à notre problème :

1. On reprend le réseau **ResNet18** avec ses poids ImageNet.
2. On **gèle** ces couches (le « backbone ») : elles servent d'**extracteur de
   caractéristiques** figé.
3. On remplace seulement la **dernière couche** (la « tête ») par une couche
   adaptée à nos 2 classes, et c'est **la seule qu'on entraîne**.

C'est l'approche dite **extraction de caractéristiques** : rapide (peu de
paramètres à apprendre) et efficace quand on a peu de données.

### 1.3 Entrée / sortie
| | Description |
|---|---|
| **Entrée** | Image RGB `224 × 224`, normalisée avec les statistiques **ImageNet** (obligatoire : le réseau a été entraîné ainsi). |
| **Sortie** | 2 *logits* (NORMAL / PNEUMONIA). |

---

## 2. Données utilisées

Identiques au Modèle A (pour une comparaison juste) : dataset
*Chest X-Ray Images (Pneumonia)*, **re-split stratifié 85/15** des `train`+`val`
d'origine → **train 4447 / val 785 / test 624**.

- Déséquilibre des classes géré par **`WeightedRandomSampler`**.
- Transforms : resize `224×224`, augmentation (flip + rotation ±10°) sur le train,
  normalisation **ImageNet** (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
- `Dataset` réutilisé : `ChestXrayDataset` (module `chest_dataset.py`), qui
  convertit chaque radio en **RGB 3 canaux**.

---

## 3. Vue technique — Architecture

Construite par la fonction `build_model_b` :

```python
net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
if freeze:                              # FREEZE_BACKBONE = True par défaut
    for param in net.parameters():
        param.requires_grad = False     # backbone gelé
net.fc = nn.Linear(net.fc.in_features, num_classes)   # nouvelle tête (512 -> 2)
```

### 3.1 Le backbone ResNet18
- Architecture profonde à **blocs résiduels** (*skip connections*) qui
  facilitent l'entraînement des réseaux profonds.
- ~**11,2 M de paramètres** au total.
- Ici **gelé** : ses poids ne bougent pas pendant l'entraînement.

### 3.2 La tête remplacée
- La couche finale d'origine (`fc`, 512 → 1000 classes ImageNet) est remplacée
  par `Linear(512 → 2)`.
- C'est la **seule** partie entraînable → seulement **~1 026 paramètres**
  (`512 × 2 + 2`) à apprendre.

### 3.3 Variante : fine-tuning complet
Mettre `FREEZE_BACKBONE = False` dégèle tout le réseau : on ré-ajuste alors
**tous** les poids (plus puissant mais beaucoup plus lent sur CPU). Par défaut on
reste en extraction de caractéristiques.

---

## 4. Vue technique — Entraînement

### 4.1 Configuration
| Élément | Valeur | Rôle |
|---|---|---|
| Perte | `CrossEntropyLoss` | Classification 2 classes. |
| Optimiseur | `Adam` (lr = `1e-3`) | **Uniquement sur les paramètres entraînables** (`p.requires_grad`), donc la tête. |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2) | Baisse le LR si la val stagne. |
| Epochs max | `15` | Le transfer learning converge vite. |
| Early stopping | patience `4` | Arrêt si la val loss ne progresse plus. |
| Batch size | `32` | — |
| `num_workers` | `8` | Lecture parallèle (8 cœurs CPU). |
| Device | CPU | — |

> Détail clé : l'optimiseur ne reçoit que
> `params = [p for p in model.parameters() if p.requires_grad]`. Avec le backbone
> gelé, seul l'apprentissage de la tête est effectué → entraînement rapide.

### 4.2 Boucle d'entraînement
Identique au Modèle A (fonctions `run_epoch` et `train_model`) :
- une passe train + une passe validation par epoch ;
- suivi de `history` (loss/accuracy train & val) ;
- conservation des **meilleurs poids** (`best_state`) sur la val loss ;
- early stopping.

Le modèle recharge ensuite ses meilleurs poids :
`model_b.load_state_dict(training_result_b["best_state"])`.

### 4.3 Suivi
Deux courbes en fin d'entraînement : **loss** et **accuracy** (train vs val).

---

## 5. Sauvegarde du modèle

Sérialisé via `torch.save` dans `models/model_b_resnet18.pt` :

```python
{
    "model_state": <best_state>,
    "history": <history>,
    "classes": ["NORMAL", "PNEUMONIA"],
    "img_size": 224,
    "arch": "resnet18",
    "freeze_backbone": True,
}
```

### Rechargement
```python
from torchvision import models
import torch

checkpoint = torch.load("models/model_b_resnet18.pt", weights_only=False)
model = models.resnet18(weights=None)                 # squelette nu
model.fc = torch.nn.Linear(model.fc.in_features, 2)   # même tête
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

---

## 6. Pour aller plus loin
- **Fine-tuning complet** (`FREEZE_BACKBONE = False`) ou dégel partiel des
  derniers blocs (`layer4`) pour gagner encore en performance, idéalement sur GPU.
- Tester d'autres backbones (DenseNet121, EfficientNet).
- Ajustement du **seuil de décision** pour affiner le compromis
  sensibilité/spécificité.
- Comparaison finale **A vs B** (cellule dédiée dans `ModelB.py`) pour justifier
  le modèle retenu.
