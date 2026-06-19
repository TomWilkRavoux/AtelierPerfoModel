# Plan — Détection de pneumonie sur radios pulmonaires (Deep Learning)

## Objectif

Concevoir, entraîner et comparer des modèles de **Deep Learning** (CNN) pour
classifier des radiographies thoraciques en `NORMAL` vs `PNEUMONIA`, puis
justifier le modèle retenu via une analyse de performance.

> Contrainte choisie : **Deep Learning uniquement** (pas de ML classique).

---

## État des lieux

- **Stack** : projet `uv` + `marimo` (notebook = `Exploration.py`) + `torch`.
- **Données** : dataset Kaggle *Chest X-Ray Images (Pneumonia)*, déjà dézippé
  dans `asset/chest_xray/`, organisé en `train/val/test` × `NORMAL/PNEUMONIA`.

### Répartition des données (à exploiter dans la présentation)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| train | 1341   | 3875      | 5216  |
| val   | 8      | 8         | 16 (trop petit) |
| test  | 234    | 390       | 624   |

### Deux problèmes à traiter
1. **Déséquilibre des classes** : ~3× plus de pneumonies que de cas normaux.
2. **Set de validation inutilisable** : 16 images seulement → à reconstruire.

---

## Phases

### Phase 0 — Setup ✅
- [x] Ajouter les dépendances manquantes : `torchvision`, `scikit-learn`,
      `matplotlib`, `pillow`.
- [x] Corriger la typo `polar` → `polars` dans `pyproject.toml`
      (l'ancien `polar` tirait des paquets parasites : pandas, nltk, pptx… retirés).
- [x] Environnement synchronisé (`uv sync`) et imports vérifiés :
      torch 2.12.1+cu130, torchvision 0.27.1+cu130, sklearn 1.9.0,
      matplotlib 3.11.0.
- [x] Détecter le device (CPU / GPU CUDA) — **CUDA non dispo → entraînement CPU**
      (codé dans le notebook, cellule device).

### Phase 1 — Exploration des données *(sujet : point 1)* ✅
- [x] Compter et visualiser la répartition des classes (tableau + barres groupées)
      → déséquilibre ~3:1 confirmé.
- [x] Afficher un échantillon d'images `NORMAL` vs `PNEUMONIA`.
- [x] Vérifier dimensions / canaux : images en niveaux de gris (mode `L`),
      tailles très variables (largeurs ~994→2338 px) → resize obligatoire.

> Tout est dans `Exploration.py` (notebook marimo, cellules Phase 1).
> Lancer en interactif : `uv run marimo edit Exploration.py`.

### Phase 2 — Préparation / transformation *(sujet : point 2, image → numérique)* ✅
- [x] **Re-split** stratifié : fusion `train`+`val` puis recoupe 85/15
      → **train 4447 / val 785 / test 624**.
- [x] `transforms` torchvision : resize 224×224, `ToTensor`, normalisation
      ImageNet (conversion RGB 3 canaux dans le `Dataset`).
- [x] **Data augmentation** sur le train (flip horizontal + rotation ±10°),
      aucune sur val/test.
- [x] Déséquilibre géré par `WeightedRandomSampler` (+ `class_weights_tensor`
      dispo pour une loss pondérée en alternative).
- [x] `Dataset` custom `ChestXrayDataset` + `DataLoader` (batch=32).
      Batch vérifié : shape `(32, 3, 224, 224)`, normalisation OK, classes
      ré-équilibrées par le sampler.

> Choix : `Dataset` custom plutôt qu'`ImageFolder` car le re-split mélange
> les dossiers `train`/`val` d'origine.

### Phase 3 — Modèles DL à comparer *(sujet : points 3–4)*
- [x] **Modèle A — CNN maison** (from scratch) : baseline DL.
      `SimpleCNN` (4 blocs Conv-BN-ReLU-MaxPool + classifieur dense),
      **~0.50 M paramètres**, sortie 2 logits. Défini dans `Exploration.py`.
- [x] **Modèle B — Transfer learning** : **ResNet18 pré-entraîné ImageNet**,
      tête `fc` remplacée. Notebook autonome `ModelB.py` couvrant prépa data +
      entraînement + sauvegarde (`models/model_b_resnet18.pt`) + évaluation +
      **comparaison A vs B**. Par défaut backbone gelé (`FREEZE_BACKBONE=True`,
      extraction de features, rapide sur CPU) ; passer à `False` pour un
      fine-tuning complet.
      ⏳ **À lancer par l'utilisateur** : cellule « LANCER L'ENTRAÎNEMENT » de
      `ModelB.py`, puis sauvegarde + cellules d'évaluation.
- [ ] *(optionnel)* **Modèle C** : 2ᵉ archi pré-entraînée pour enrichir la
      comparaison.

> Stratégie : finaliser entièrement le Modèle A ici (Phases 4–5), puis ouvrir
> un 2ᵉ notebook pour le Modèle B et comparer les deux.

### Phase 4 — Entraînement *(Modèle A)* ✅ *(code prêt, à lancer)*
- [x] Boucle train / val, `CrossEntropyLoss`, optimiseur **Adam**,
      `ReduceLROnPlateau`.
- [x] Early stopping (patience 5) + conservation/rechargement des meilleurs poids.
- [x] Courbes loss / accuracy par epoch.
- [x] **Sauvegarde** du modèle → `models/model_a_cnn.pt` (state + history +
      classes + img_size). `models/` ajouté au `.gitignore`.
- [x] `NUM_WORKERS = 8` (8 cœurs CPU), `NUM_EPOCHS = 30`.
- [ ] **À faire par l'utilisateur** : lancer la cellule « LANCER L'ENTRAÎNEMENT »
      (CPU → long), puis les cellules courbes + sauvegarde.

> Pipeline validé par un smoke-test (run_epoch / train_model / save+reload OK).

### Phase 5 — Évaluation & choix *(sujet : point 5)*
- [x] Évaluation **autonome** : modèle rechargé depuis `models/model_a_cnn.pt`
      (indépendante de l'entraînement).
- [x] **Métriques** sur le test (624 img) : accuracy, précision, **rappel /
      sensibilité**, spécificité, F1, **ROC-AUC**.
- [x] **Matrice de confusion** + **courbe ROC** + lecture clinique.
- [ ] Tableau comparatif **Modèle A vs Modèle B** → justifier le retenu
      *(à faire une fois le Modèle B entraîné)*.

**Résultats Modèle A (test)** : acc 0.729 · précision 0.699 · **sensibilité
0.995** · spécificité 0.286 · F1 0.821 · **ROC-AUC 0.919**.
Confusion `[[TN 67, FP 167],[FN 2, TP 388]]`.

> Constat : sensibilité quasi parfaite (2 pneumonies manquées) mais spécificité
> faible (167 fausses alertes). AUC 0.92 élevé → le modèle sépare bien les
> classes ; le **seuil 0.5 est biaisé** (effet du sampler). Piste : tuning du
> seuil de décision pour remonter la spécificité.

### Extension — Classification 3 classes (sain / bactérien / viral)
- [x] Notebook `PneumoniaType.py` : **recharge le Modèle A**, remplace la tête
      `Linear(128→2)` par `Linear(128→3)`, fine-tune tout le réseau.
- [x] Labels viral/bactérien déduits des **noms de fichiers**
      (`_bacteria_` / `_virus_`). Architecture sortie dans `model_a_cnn.py`.
- [x] Répartition : NORMAL 1341 / BACTÉRIENNE 2530 / VIRALE 1345 (train+val).
- [x] Évaluation multi-classes : accuracy, F1 macro, précision/rappel/F1 par
      classe, matrice de confusion 3×3. Sauvegarde →
      `models/model_c_pneumonia_type.pt`.
      ⏳ **À lancer par l'utilisateur**.

> ⚠️ Tâche difficile : viral vs bactérien est peu séparable sur radio
> (~60–75 % dans la littérature). Bon point de discussion.

### Phase 6 — Présentation ✅
- [x] Contenu des slides rédigé dans `slides.md` (prêt à importer dans Gamma).
      Couvre : sujet & données, méthodologie, modèles testés, modèle retenu
      (B), justification par les métriques, bonus 3 classes, limites.
      Emplacements `📸 CAPTURE` pour insérer les screenshots de métriques.

---

## Livrables
- `Exploration.py` (notebook marimo) couvrant les phases 1 → 5.
- `PLAN.md` (ce fichier).
- Support de présentation (phase 6).
