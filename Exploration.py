import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import torch
    from pathlib import Path
    from collections import Counter
    from PIL import Image
    import matplotlib.pyplot as plt
    import random

    return Image, Path, mo, np, pl, plt, random, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Détection de pneumonie — Exploration des données

    **Phase 1 du plan** : récupérer et explorer les radiographies thoraciques
    (`NORMAL` vs `PNEUMONIA`) avant toute modélisation.

    Objectifs :
    - compter la répartition des classes (mettre en évidence le déséquilibre) ;
    - visualiser des échantillons d'images ;
    - inspecter les dimensions / canaux des images.
    """)
    return


@app.cell
def _(mo, torch):
    # --- Device : CUDA si dispo, sinon CPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo.md(f"**Device utilisé pour l'entraînement : `{device}`**")
    return (device,)


@app.cell
def _(Path, mo):
    # --- Localisation du dataset ---
    DATA_DIR = Path("asset/chest_xray")
    SPLITS = ["train", "val", "test"]
    CLASSES = ["NORMAL", "PNEUMONIA"]
    IMG_EXTS = {".jpeg", ".jpg", ".png"}

    assert DATA_DIR.exists(), f"Dossier introuvable : {DATA_DIR.resolve()}"
    mo.md(f"Dataset : `{DATA_DIR.resolve()}`")
    return CLASSES, DATA_DIR, IMG_EXTS, SPLITS


@app.cell
def _(CLASSES, DATA_DIR, IMG_EXTS, Path, SPLITS):
    def list_images(split: str, cls: str) -> list[Path]:
        """Liste les fichiers image d'un split/classe donné."""
        folder = DATA_DIR / split / cls
        if not folder.exists():
            return []
        return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]

    # Index complet {(split, classe): [chemins...]}
    images_index = {
        (split, cls): list_images(split, cls)
        for split in SPLITS
        for cls in CLASSES
    }
    return (images_index,)


@app.cell
def _(CLASSES, SPLITS, images_index, mo, pl):
    # --- Tableau de répartition des classes ---
    _rows = []
    for _split in SPLITS:
        _row = {"split": _split}
        _total = 0
        for _cls in CLASSES:
            _n = len(images_index[(_split, _cls)])
            _row[_cls] = _n
            _total += _n
        _row["TOTAL"] = _total
        _rows.append(_row)

    counts_df = pl.DataFrame(_rows)
    mo.vstack([mo.md("## Répartition des classes"), counts_df])
    return


@app.cell
def _(CLASSES, SPLITS, images_index, mo, np, plt):
    # --- Graphe en barres groupées de la répartition ---
    _x = np.arange(len(SPLITS))
    _width = 0.35

    _fig_dist, _ax_dist = plt.subplots(figsize=(7, 4))
    for _i, _cls in enumerate(CLASSES):
        _values = [len(images_index[(_split, _cls)]) for _split in SPLITS]
        _bars = _ax_dist.bar(_x + (_i - 0.5) * _width, _values, _width, label=_cls)
        _ax_dist.bar_label(_bars, padding=2, fontsize=8)

    _ax_dist.set_xticks(_x)
    _ax_dist.set_xticklabels(SPLITS)
    _ax_dist.set_ylabel("Nombre d'images")
    _ax_dist.set_title("Répartition des classes par split")
    _ax_dist.legend()
    _fig_dist.tight_layout()

    mo.vstack(
        [
            mo.md(
                "### Observations\n"
                "- **Déséquilibre** : ~3× plus de `PNEUMONIA` que de `NORMAL` "
                "dans le train → à compenser (loss pondérée / sampler).\n"
                "- **Validation minuscule** (16 images) → à reconstruire en Phase 2."
            ),
            _fig_dist,
        ]
    )
    return


@app.cell
def _(CLASSES, Image, images_index, mo, plt, random):
    # --- Échantillon d'images par classe ---
    random.seed(42)
    _n_per_class = 4

    _fig_samples, _axes = plt.subplots(
        len(CLASSES), _n_per_class, figsize=(3 * _n_per_class, 3 * len(CLASSES))
    )
    for _r, _cls in enumerate(CLASSES):
        _paths = images_index[("train", _cls)]
        _picks = random.sample(_paths, min(_n_per_class, len(_paths)))
        for _c in range(_n_per_class):
            _ax = _axes[_r][_c]
            _ax.axis("off")
            if _c < len(_picks):
                _img = Image.open(_picks[_c]).convert("L")  # niveaux de gris
                _ax.imshow(_img, cmap="gray")
                _ax.set_title(f"{_cls}\n{_img.size}", fontsize=8)
    _fig_samples.suptitle("Échantillon d'images (train)", fontsize=13)
    _fig_samples.tight_layout()

    mo.vstack([mo.md("## Échantillon d'images"), _fig_samples])
    return


@app.cell
def _(Image, SPLITS, images_index, mo, np, pl):
    # --- Statistiques sur les dimensions et les canaux ---
    # On échantillonne pour rester rapide (pas besoin de tout ouvrir).
    from collections import Counter as _Counter

    _sample_paths = []
    for _split in SPLITS:
        for _key, _paths in images_index.items():
            if _key[0] == _split:
                _sample_paths.extend(_paths[:100])

    _widths, _heights, _modes = [], [], []
    for _p in _sample_paths:
        with Image.open(_p) as _im:
            _widths.append(_im.width)
            _heights.append(_im.height)
            _modes.append(_im.mode)

    _w = np.array(_widths)
    _h = np.array(_heights)

    dim_summary = pl.DataFrame(
        {
            "stat": ["min", "médiane", "moyenne", "max"],
            "largeur": [
                float(_w.min()),
                float(np.median(_w)),
                round(float(_w.mean()), 1),
                float(_w.max()),
            ],
            "hauteur": [
                float(_h.min()),
                float(np.median(_h)),
                round(float(_h.mean()), 1),
                float(_h.max()),
            ],
        }
    )

    _modes_count = dict(_Counter(_modes))

    mo.vstack(
        [
            mo.md(
                "## Dimensions & canaux\n"
                f"Échantillon analysé : **{len(_sample_paths)} images**.\n\n"
                f"Modes PIL rencontrés : `{_modes_count}` "
                "(radios en niveaux de gris ; quelques images peuvent être RGB).\n\n"
                "Tailles très variables → **resize obligatoire** (ex. 224×224) "
                "et conversion en un nombre de canaux homogène en Phase 2."
            ),
            dim_summary,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Phase 2 — Préparation des données

    Transformer les images en tenseurs exploitables par un CNN :
    - **re-split stratifié** (le `val` d'origine ne fait que 16 images) ;
    - **transforms** torchvision : resize 224×224 + normalisation ImageNet ;
    - **data augmentation** sur le train uniquement ;
    - **gestion du déséquilibre** via `WeightedRandomSampler` ;
    - construction des `Dataset` / `DataLoader`.
    """)
    return


@app.cell
def _(CLASSES, images_index, mo, pl):
    # --- Re-split stratifié : on fusionne train+val puis on recoupe 85/15 ---
    from sklearn.model_selection import train_test_split

    LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    # (chemin, label) — les compréhensions ne polluent pas le scope de la cellule
    combined_samples = [
        (path, LABEL_MAP[cls])
        for split in ("train", "val")
        for cls in CLASSES
        for path in images_index[(split, cls)]
    ]
    test_samples = [
        (path, LABEL_MAP[cls])
        for cls in CLASSES
        for path in images_index[("test", cls)]
    ]

    train_samples, val_samples = train_test_split(
        combined_samples,
        test_size=0.15,
        stratify=[label for _, label in combined_samples],
        random_state=42,
    )

    split_summary = pl.DataFrame(
        {
            "split": ["train", "val", "test"],
            "NORMAL": [
                sum(1 for _, label in train_samples if label == 0),
                sum(1 for _, label in val_samples if label == 0),
                sum(1 for _, label in test_samples if label == 0),
            ],
            "PNEUMONIA": [
                sum(1 for _, label in train_samples if label == 1),
                sum(1 for _, label in val_samples if label == 1),
                sum(1 for _, label in test_samples if label == 1),
            ],
            "TOTAL": [len(train_samples), len(val_samples), len(test_samples)],
        }
    )
    mo.vstack(
        [mo.md("## Re-split stratifié (train+val → 85/15)"), split_summary]
    )
    return test_samples, train_samples, val_samples


@app.cell
def _():
    # --- Transforms : resize 224 + normalisation ImageNet ---
    from torchvision import transforms

    IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Train : augmentation légère (on ne déforme pas trop une radio médicale)
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Validation / test : pas d'augmentation
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return IMG_SIZE, eval_transform, train_transform


@app.cell
def _():
    # --- Dataset custom : (chemin, label) -> tenseur RGB normalisé ---
    # Importé depuis chest_dataset.py (et non défini ici) pour rester picklable
    # par les workers du DataLoader (num_workers > 0).
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from chest_dataset import ChestXrayDataset

    return ChestXrayDataset, DataLoader, WeightedRandomSampler


@app.cell
def _(
    ChestXrayDataset,
    eval_transform,
    mo,
    test_samples,
    train_samples,
    train_transform,
    val_samples,
):
    # --- Instanciation des datasets ---
    train_ds = ChestXrayDataset(train_samples, train_transform)
    val_ds = ChestXrayDataset(val_samples, eval_transform)
    test_ds = ChestXrayDataset(test_samples, eval_transform)
    mo.md(
        "## Datasets\n"
        f"- train : {len(train_ds)} images (avec augmentation)\n"
        f"- val : {len(val_ds)} images\n"
        f"- test : {len(test_ds)} images"
    )
    return test_ds, train_ds, val_ds


@app.cell
def _(WeightedRandomSampler, mo, torch, train_samples):
    # --- Gestion du déséquilibre des classes ---
    train_class_counts = [
        sum(1 for _, label in train_samples if label == 0),
        sum(1 for _, label in train_samples if label == 1),
    ]
    # Poids par classe = inverse de la fréquence
    class_weights = [
        len(train_samples) / (2 * count) for count in train_class_counts
    ]
    # Poids par échantillon, pour le sampler
    sample_weights = [class_weights[label] for _, label in train_samples]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    # Alternative : tensor de poids pour une loss pondérée
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    mo.md(
        "## Gestion du déséquilibre\n"
        f"- Comptage train : NORMAL={train_class_counts[0]}, "
        f"PNEUMONIA={train_class_counts[1]}\n"
        f"- Poids par classe : {[round(w, 3) for w in class_weights]}\n"
        "- Stratégie retenue : `WeightedRandomSampler` (les 2 classes seront "
        "tirées de façon ~équilibrée). `class_weights_tensor` reste dispo pour "
        "une loss pondérée en alternative."
    )
    return (train_sampler,)


@app.cell
def _(DataLoader, mo, test_ds, train_ds, train_sampler, val_ds):
    # --- DataLoaders ---
    BATCH_SIZE = 32
    NUM_WORKERS = 8  # 8 cœurs CPU dispo → lecture des images en parallèle

    # Le sampler gère déjà le tirage → pas de shuffle sur le train
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    mo.md(
        f"## DataLoaders (batch={BATCH_SIZE})\n"
        f"- train : {len(train_loader)} batches\n"
        f"- val : {len(val_loader)} batches\n"
        f"- test : {len(test_loader)} batches"
    )
    return test_loader, train_loader, val_loader


@app.cell
def _(mo, train_loader):
    # --- Vérification d'un batch ---
    batch_images, batch_labels = next(iter(train_loader))
    mo.md(
        "## Vérification d'un batch\n"
        f"- Images : shape `{tuple(batch_images.shape)}`, "
        f"dtype `{batch_images.dtype}`\n"
        f"- Valeurs (post-normalisation) min/max : "
        f"{batch_images.min():.2f} / {batch_images.max():.2f}\n"
        f"- Répartition des labels du batch : "
        f"{batch_labels.sum().item()} pneumonies / {len(batch_labels)}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Phase 3 — Modèle A : CNN maison (*from scratch*)

    Premier modèle Deep Learning, entraîné de zéro, qui servira de
    **baseline**. Architecture classique : empilement de blocs
    `Conv → BatchNorm → ReLU → MaxPool`, puis un classifieur dense.

    > Le **Modèle B** (transfer learning ResNet/DenseNet) sera traité dans un
    > notebook séparé, pour comparer ensuite les deux approches.
    """)
    return


@app.cell
def _():
    # --- Architecture du CNN maison (Modèle A) ---
    from torch import nn

    class SimpleCNN(nn.Module):
        """CNN de base pour la classification binaire NORMAL / PNEUMONIA.

        Entrée : (B, 3, 224, 224) -> sortie : (B, 2) logits.
        """

        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                self._conv_block(3, 32),    # 224 -> 112
                self._conv_block(32, 64),   # 112 -> 56
                self._conv_block(64, 128),  # 56 -> 28
                self._conv_block(128, 128),  # 28 -> 14
            )
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(128 * 4 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        @staticmethod
        def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            return self.classifier(x)

    return SimpleCNN, nn


@app.cell
def _(SimpleCNN, device, mo, torch):
    # --- Instanciation + vérification d'une passe avant ---
    model_a = SimpleCNN(num_classes=2).to(device)

    model_a_params = sum(p.numel() for p in model_a.parameters())

    # Passe avant sur un batch factice pour valider les dimensions
    dummy_batch = torch.randn(2, 3, 224, 224, device=device)
    model_a.eval()
    with torch.no_grad():
        dummy_out_shape = tuple(model_a(dummy_batch).shape)

    mo.md(
        "## Modèle A instancié\n"
        f"- Paramètres : **{model_a_params:,}** (~{model_a_params / 1e6:.2f} M)\n"
        f"- Sortie pour un batch de 2 images : `{dummy_out_shape}` (2 logits/image)\n"
        f"- Device : `{device}`"
    )
    return (model_a,)


@app.cell
def _(mo):
    mo.md(r"""
    # Phase 4 — Entraînement du Modèle A

    Boucle d'entraînement / validation avec :
    - perte `CrossEntropyLoss`, optimiseur **Adam** ;
    - `ReduceLROnPlateau` (baisse le LR si la val stagne) ;
    - **early stopping** + conservation des meilleurs poids ;
    - suivi des courbes loss / accuracy ;
    - **sauvegarde du modèle** entraîné.

    > ⚠️ Entraînement sur **CPU** : lancer manuellement la cellule
    > « Lancer l'entraînement » et la laisser tourner.
    """)
    return


@app.cell
def _(nn):
    # --- Hyperparamètres + critère ---
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    PATIENCE = 5  # epochs sans amélioration avant early stopping

    criterion = nn.CrossEntropyLoss()  # le sampler équilibre déjà les classes
    return LEARNING_RATE, NUM_EPOCHS, PATIENCE, criterion


@app.cell
def _(device, torch):
    # --- Une epoch (train si optimizer fourni, sinon évaluation) ---
    def run_epoch(model, loader, criterion, optimizer=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        running_loss, correct, seen = 0.0, 0, 0
        with torch.set_grad_enabled(is_train):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                if is_train:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if is_train:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                seen += labels.size(0)
        return running_loss / seen, correct / seen

    return (run_epoch,)


@app.cell
def _(run_epoch, torch):
    # --- Boucle complète avec early stopping ---
    import copy
    import time

    def train_model(model, train_loader, val_loader, criterion, num_epochs, lr, patience):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                marker = "  <- best"
            else:
                epochs_no_improve += 1

            print(
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}{marker}"
            )
            if epochs_no_improve >= patience:
                print(f"Early stopping : pas d'amélioration depuis {patience} epochs.")
                break

        print(f"Terminé en {time.time() - t0:.0f}s | meilleure val loss : {best_val_loss:.4f}")
        return {
            "history": history,
            "best_state": best_state,
            "best_val_loss": best_val_loss,
        }

    return (train_model,)


@app.cell
def _(
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    criterion,
    mo,
    model_a,
    train_loader,
    train_model,
    val_loader,
):
    # === LANCER L'ENTRAÎNEMENT (cellule lourde — exécution manuelle) ===
    training_result = train_model(
        model_a,
        train_loader,
        val_loader,
        criterion,
        NUM_EPOCHS,
        LEARNING_RATE,
        PATIENCE,
    )
    # Recharge les meilleurs poids dans le modèle
    model_a.load_state_dict(training_result["best_state"])
    mo.md(
        "## Entraînement terminé\n"
        f"- Meilleure val loss : **{training_result['best_val_loss']:.4f}**\n"
        f"- Epochs effectuées : {len(training_result['history']['train_loss'])}\n"
        "- Les meilleurs poids ont été rechargés dans `model_a`."
    )
    return (training_result,)


@app.cell
def _(mo, plt, training_result):
    # --- Courbes loss / accuracy ---
    history_a = training_result["history"]
    epochs_axis = range(1, len(history_a["train_loss"]) + 1)

    fig_curves, axes_curves = plt.subplots(1, 2, figsize=(12, 4))
    axes_curves[0].plot(epochs_axis, history_a["train_loss"], label="train")
    axes_curves[0].plot(epochs_axis, history_a["val_loss"], label="val")
    axes_curves[0].set_title("Loss")
    axes_curves[0].set_xlabel("epoch")
    axes_curves[0].legend()

    axes_curves[1].plot(epochs_axis, history_a["train_acc"], label="train")
    axes_curves[1].plot(epochs_axis, history_a["val_acc"], label="val")
    axes_curves[1].set_title("Accuracy")
    axes_curves[1].set_xlabel("epoch")
    axes_curves[1].legend()
    fig_curves.tight_layout()

    fig_curves.savefig('courbes_apprentissage.png', dpi=200, bbox_inches='tight')

    mo.vstack([mo.md("## Courbes d'apprentissage"), fig_curves])
    return


@app.cell
def _(IMG_SIZE, Path, mo, torch, training_result):
    # --- Sauvegarde du modèle entraîné ---
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_A_PATH = MODEL_DIR / "model_a_cnn.pt"

    torch.save(
        {
            "model_state": training_result["best_state"],
            "history": training_result["history"],
            "classes": ["NORMAL", "PNEUMONIA"],
            "img_size": IMG_SIZE,
            "arch": "SimpleCNN",
        },
        MODEL_A_PATH,
    )
    mo.md(f"## Modèle sauvegardé\n`{MODEL_A_PATH.resolve()}`")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Phase 5 — Évaluation du Modèle A

    Évaluation **sur le set de test** (jamais vu pendant l'entraînement).
    Le modèle est **rechargé depuis le disque** (`models/model_a_cnn.pt`),
    donc cette phase est indépendante de l'entraînement : après un reload du
    kernel, inutile de ré-entraîner.

    Métriques adaptées au contexte médical (on veut surtout **ne pas rater
    une pneumonie**) : matrice de confusion, **rappel / sensibilité**,
    précision, spécificité, F1-score, ROC-AUC.
    """)
    return


@app.cell
def _(Path, SimpleCNN, device, mo, torch):
    # --- Rechargement du modèle entraîné depuis le disque ---
    EVAL_MODEL_PATH = Path("models/model_a_cnn.pt")
    assert EVAL_MODEL_PATH.exists(), (
        f"Modèle introuvable : {EVAL_MODEL_PATH}. "
        "Lance d'abord l'entraînement + la cellule de sauvegarde."
    )

    eval_checkpoint = torch.load(
        EVAL_MODEL_PATH, map_location=device, weights_only=False
    )
    model_a_eval = SimpleCNN(num_classes=2).to(device)
    model_a_eval.load_state_dict(eval_checkpoint["model_state"])
    model_a_eval.eval()

    mo.md(
        "## Modèle rechargé\n"
        f"- Source : `{EVAL_MODEL_PATH}`\n"
        f"- Architecture : `{eval_checkpoint['arch']}`\n"
        f"- Epochs entraînées : {len(eval_checkpoint['history']['train_loss'])}"
    )
    return (model_a_eval,)


@app.cell
def _(device, model_a_eval, test_loader, torch):
    # --- Prédictions sur le set de test ---
    def collect_predictions(model, loader):
        labels_all, preds_all, probs_all = [], [], []
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images.to(device))
                probs_pneumonia = torch.softmax(logits, dim=1)[:, 1]
                labels_all.extend(labels.tolist())
                preds_all.extend(logits.argmax(1).cpu().tolist())
                probs_all.extend(probs_pneumonia.cpu().tolist())
        return labels_all, preds_all, probs_all

    y_true, y_pred, y_prob = collect_predictions(model_a_eval, test_loader)
    return y_pred, y_prob, y_true


@app.cell
def _(CLASSES, mo, pl, y_pred, y_prob, y_true):
    # --- Calcul des métriques ---
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        classification_report,
    )

    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()  # classe positive = PNEUMONIA (1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)  # sensibilité
    spec = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cr = classification_report(y_true, y_pred, target_names=CLASSES)

    metrics_df = pl.DataFrame(
        {
            "métrique": [
                "Accuracy",
                "Précision (PNEUMONIA)",
                "Rappel / Sensibilité",
                "Spécificité",
                "F1-score",
                "ROC-AUC",
            ],
            "valeur": [
                round(float(acc), 4),
                round(float(prec), 4),
                round(float(rec), 4),
                round(float(spec), 4),
                round(float(f1), 4),
                round(float(auc), 4),
            ],
        }
    )

    print(cr)
    mo.vstack([mo.md("## Métriques (set de test)"), metrics_df])
    return auc, conf_matrix, fn, fp, tn, tp


@app.cell
def _(CLASSES, conf_matrix, mo, plt):
    # --- Matrice de confusion ---
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    im_cm = ax_cm.imshow(conf_matrix, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_xticklabels(CLASSES)
    ax_cm.set_yticks([0, 1])
    ax_cm.set_yticklabels(CLASSES)
    ax_cm.set_xlabel("Prédit")
    ax_cm.set_ylabel("Réel")
    ax_cm.set_title("Matrice de confusion (test)")
    for cm_row in range(conf_matrix.shape[0]):
        for cm_col in range(conf_matrix.shape[1]):
            ax_cm.text(
                cm_col,
                cm_row,
                str(conf_matrix[cm_row, cm_col]),
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )
    fig_cm.colorbar(im_cm)
    fig_cm.tight_layout()
    mo.vstack([mo.md("## Matrice de confusion"), fig_cm])
    return


@app.cell
def _(auc, mo, plt, y_prob, y_true):
    # --- Courbe ROC ---
    from sklearn.metrics import roc_curve

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="grey")
    ax_roc.set_xlabel("Taux de faux positifs")
    ax_roc.set_ylabel("Taux de vrais positifs (sensibilité)")
    ax_roc.set_title("Courbe ROC (test)")
    ax_roc.legend()
    fig_roc.tight_layout()
    mo.vstack([mo.md("## Courbe ROC"), fig_roc])
    return


@app.cell
def _(fn, fp, mo, tn, tp):
    # --- Lecture clinique des résultats ---
    mo.md(
        "## Analyse\n"
        f"- **Vrais positifs (pneumonies détectées)** : {tp}\n"
        f"- **Faux négatifs (pneumonies manquées)** : {fn} ← le plus grave en "
        "contexte médical\n"
        f"- **Faux positifs (fausses alertes)** : {fp}\n"
        f"- **Vrais négatifs (sains bien classés)** : {tn}\n\n"
        "En dépistage, on privilégie un **rappel/sensibilité élevé** : mieux "
        "vaut une fausse alerte (faux positif, levée par un médecin) qu'une "
        "pneumonie non détectée (faux négatif). C'est l'axe de comparaison "
        "principal avec le futur Modèle B."
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
