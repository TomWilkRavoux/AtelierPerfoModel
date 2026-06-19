# Modèle C — Fine-tuning du Modèle A en 3 classes : construction & entraînement

> Documentation du **Modèle C** : ré-utilisation du Modèle A pour distinguer le
> **type** de pneumonie (bactérien / viral) en plus de sain / infecté. Le code se
> trouve dans le notebook marimo [`PneumoniaType.py`](../PneumoniaType.py).
>
> Voir aussi : [analyse des métriques du Modèle C](analyse_metrics_modele_c.md).

---

## 1. Vue fonctionnelle

### 1.1 Objectif
Passer d'une classification **binaire** (NORMAL / PNEUMONIA) à une classification
à **3 classes** :

- `0` = **NORMAL** (poumon sain)
- `1` = **PNEUMONIE BACTÉRIENNE**
- `2` = **PNEUMONIE VIRALE**

### 1.2 Principe : fine-tuning par transfert interne
Le Modèle A a déjà appris à « lire » des radios pulmonaires. Plutôt que de
repartir de zéro, on **réutilise tout son savoir** :

1. On **recharge** les poids du Modèle A (`models/model_a_cnn.pt`).
2. On **garde** les couches de convolution **et** la première couche dense
   (extraction de caractéristiques déjà pertinente).
3. On **remplace uniquement la dernière couche** `Linear(128 → 2)` par
   `Linear(128 → 3)` (la nouvelle tête, initialisée au hasard).
4. On **fine-tune tout le réseau** avec un *learning rate* plus faible, puisqu'on
   part de poids déjà bons.

### 1.3 Origine des étiquettes
Le dataset ne fournit pas de dossier « bactérien » / « viral », mais l'info est
**dans le nom des fichiers** : `..._bacteria_...` ou `..._virus_...`. Une fonction
`label_from_path` déduit la classe à partir du nom.

### 1.4 Difficulté intrinsèque
Distinguer **viral vs bactérien** sur une simple radio est **difficile**, même
pour un radiologue (la littérature plafonne souvent vers 60–75 %). On s'attend à
de bonnes perfs sur NORMAL et à plus de confusion entre les deux types.

---

## 2. Données utilisées

| Classe | train + val | test |
|---|---|---|
| NORMAL (0) | 1349 | 234 |
| BACTÉRIENNE (1) | 2538 | 242 |
| VIRALE (2) | 1345 | 148 |

- **Re-split stratifié 85/15** des `train`+`val` d'origine, stratifié sur les
  **3 classes**.
- Transforms identiques aux autres modèles (resize 224, augmentation train,
  normalisation ImageNet).
- Déséquilibre géré par **`WeightedRandomSampler`** recalculé pour 3 classes
  (poids = `len(train) / (3 × effectif_classe)`).

---

## 3. Vue technique — Architecture

### 3.1 Rechargement + bascule de tête
```python
from model_a_cnn import SimpleCNN          # architecture exportée en module

base_model = SimpleCNN(num_classes=2)
checkpoint_a = torch.load("models/model_a_cnn.pt", weights_only=False)
base_model.load_state_dict(checkpoint_a["model_state"])   # poids 2 classes

in_features = base_model.classifier[-1].in_features        # 128
base_model.classifier[-1] = torch.nn.Linear(in_features, 3)  # 2 -> 3 classes
```

- Le reste du réseau (4 blocs Conv-BN-ReLU-MaxPool + `AdaptiveAvgPool2d(4×4)` +
  `Linear(2048 → 128)`) est **conservé avec ses poids appris**.
- Seule la sortie change : `Linear(128 → 3)`.
- L'architecture `SimpleCNN` est importée depuis `model_a_cnn.py` pour
  **garantir la correspondance exacte** avec les poids sauvegardés.

### 3.2 Paramètres entraînés
On **ne gèle rien** : tout le réseau (~0,5 M de paramètres) est fine-tuné. Les
couches reprises s'ajustent finement, la nouvelle tête s'apprend entièrement.

---

## 4. Vue technique — Entraînement

### 4.1 Configuration
| Élément | Valeur | Rôle |
|---|---|---|
| Perte | `CrossEntropyLoss` | Classification 3 classes. |
| Optimiseur | `Adam` (lr = **`5e-4`**) | LR plus bas que le Modèle A (`1e-3`) car on part de poids déjà appris : on ajuste sans tout casser. |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2) | — |
| Epochs max | `25` | — |
| Early stopping | patience `5` | — |
| Batch size | `32` | — |
| `num_workers` | `8` | — |

### 4.2 Boucle d'entraînement
Mêmes fonctions `run_epoch` / `train_model` que les autres modèles :
- passes train + val par epoch, suivi de `history` ;
- conservation des meilleurs poids sur la val loss + early stopping ;
- rechargement des meilleurs poids à la fin
  (`model_c.load_state_dict(training_result_c["best_state"])`).

---

## 5. Évaluation (spécifique au multi-classes)

Les métriques diffèrent du binaire :
- **Accuracy globale** et **F1 macro** (moyenne des F1 par classe, à poids égal —
  ne favorise pas les classes nombreuses) ;
- **rapport par classe** : précision / rappel / F1 / support
  (`classification_report`) ;
- **matrice de confusion 3×3** pour visualiser **quelles classes se confondent**
  (typiquement NORMAL ↔ VIRALE et BACT. ↔ VIRALE).

---

## 6. Sauvegarde du modèle

Sérialisé dans `models/model_c_pneumonia_type.pt` :

```python
{
    "model_state": <best_state>,
    "history": <history>,
    "classes": ["NORMAL", "BACTERIENNE", "VIRALE"],
    "img_size": 224,
    "arch": "SimpleCNN-3classes",
    "from": "model_a_cnn.pt",
}
```

### Rechargement
```python
from model_a_cnn import SimpleCNN
import torch

checkpoint = torch.load("models/model_c_pneumonia_type.pt", weights_only=False)
model = SimpleCNN(num_classes=2)
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 3)
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

---

## 7. Pour aller plus loin
- **Repartir du Modèle B** (ResNet18) plutôt que du Modèle A : de meilleures
  features devraient améliorer surtout la distinction bactérien/viral.
- Remplacer le `WeightedRandomSampler` par une **loss pondérée** pour réduire le
  biais « VIRALE par défaut ».
- Garder le binaire NORMAL/PNEUMONIA si la priorité est le **dépistage**, le type
  n'étant qu'une indication secondaire.
