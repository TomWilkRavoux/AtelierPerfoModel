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
    import matplotlib.pyplot as plt

    return Path, mo, pl, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Fine-tuning du Modèle A → 3 classes (sain / bactérien / viral)

    On **recharge le Modèle A** (entraîné en 2 classes NORMAL/PNEUMONIA), on
    garde tout le réseau appris et on **remplace seulement la dernière
    couche** par une sortie à **3 classes** :

    - `0` = **NORMAL**
    - `1` = **PNEUMONIE BACTÉRIENNE**
    - `2` = **PNEUMONIE VIRALE**

    Les labels viral/bactérien sont déduits du **nom des fichiers**
    (`..._bacteria_...` / `..._virus_...`).

    > ⚠️ Distinguer viral vs bactérien sur une radio est **difficile** (la
    > littérature plafonne souvent vers 60–75 %). On s'attend à de bonnes
    > perfs sur NORMAL et à plus de confusion entre les deux types de
    > pneumonie : c'est un point d'analyse intéressant.
    """)
    return


@app.cell
def _(mo, torch):
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo.md(f"**Device : `{device}`**")
    return (device,)


@app.cell
def _(Path, mo, pl):
    # --- Construction des échantillons 3 classes (labels déduits des noms) ---
    DATA_DIR = Path("asset/chest_xray")
    IMG_EXTS = {".jpeg", ".jpg", ".png"}
    CLASS_NAMES = ["NORMAL", "BACTERIENNE", "VIRALE"]

    def label_from_path(path, folder_class):
        """0=NORMAL, 1=bactérienne, 2=virale ; None si indéterminé."""
        if folder_class == "NORMAL":
            return 0
        name = path.name.lower()
        if "bacteria" in name:
            return 1
        if "virus" in name:
            return 2
        return None

    def build_samples(split):
        out = []
        for folder_class in ("NORMAL", "PNEUMONIA"):
            folder = DATA_DIR / split / folder_class
            if not folder.exists():
                continue
            for path in folder.iterdir():
                if path.suffix.lower() not in IMG_EXTS:
                    continue
                label = label_from_path(path, folder_class)
                if label is not None:
                    out.append((path, label))
        return out

    combined_samples = build_samples("train") + build_samples("val")
    test_samples = build_samples("test")

    counts_table = pl.DataFrame(
        {
            "classe": CLASS_NAMES,
            "train+val": [
                sum(1 for _, lab in combined_samples if lab == k) for k in range(3)
            ],
            "test": [
                sum(1 for _, lab in test_samples if lab == k) for k in range(3)
            ],
        }
    )
    mo.vstack([mo.md("## Répartition des 3 classes"), counts_table])
    return CLASS_NAMES, combined_samples, test_samples


@app.cell
def _(combined_samples, mo, pl):
    # --- Re-split stratifié sur les 3 classes ---
    from sklearn.model_selection import train_test_split

    train_samples, val_samples = train_test_split(
        combined_samples,
        test_size=0.15,
        stratify=[label for _, label in combined_samples],
        random_state=42,
    )
    split_summary = pl.DataFrame(
        {
            "split": ["train", "val"],
            "total": [len(train_samples), len(val_samples)],
        }
    )
    mo.vstack([mo.md("## Re-split (85/15)"), split_summary])
    return train_samples, val_samples


@app.cell
def _():
    # --- Transforms : resize 224 + normalisation ImageNet ---
    from torchvision import transforms

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return eval_transform, train_transform


@app.cell
def _(
    eval_transform,
    mo,
    test_samples,
    train_samples,
    train_transform,
    val_samples,
):
    # --- Datasets + DataLoaders + sampler pondéré (3 classes) ---
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from chest_dataset import ChestXrayDataset

    train_ds = ChestXrayDataset(train_samples, train_transform)
    val_ds = ChestXrayDataset(val_samples, eval_transform)
    test_ds = ChestXrayDataset(test_samples, eval_transform)

    class_counts = [sum(1 for _, lab in train_samples if lab == k) for k in range(3)]
    class_weights = [len(train_samples) / (3 * c) for c in class_counts]
    sample_weights = [class_weights[lab] for _, lab in train_samples]
    train_sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    BATCH_SIZE = 32
    NUM_WORKERS = 8
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    mo.md(
        f"DataLoaders prêts — train {len(train_loader)} / val {len(val_loader)} / "
        f"test {len(test_loader)} batches | comptage train {class_counts}"
    )
    return test_loader, train_loader, val_loader


@app.cell
def _(device, mo, torch):
    # --- Rechargement du Modèle A + bascule de la tête en 3 classes ---
    from model_a_cnn import SimpleCNN

    base_model = SimpleCNN(num_classes=2)
    checkpoint_a = torch.load(
        "models/model_a_cnn.pt", map_location=device, weights_only=False
    )
    base_model.load_state_dict(checkpoint_a["model_state"])

    # On garde convolutions + 1re couche dense, on remplace la sortie 2 -> 3
    in_features = base_model.classifier[-1].in_features
    base_model.classifier[-1] = torch.nn.Linear(in_features, 3)

    model_c = base_model.to(device)
    trainable = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
    mo.md(
        "## Modèle rechargé et adapté\n"
        "- Poids du Modèle A chargés (convolutions + dense conservées)\n"
        f"- Nouvelle tête : `Linear({in_features} → 3)`\n"
        f"- Fine-tuning de **tout** le réseau ({trainable:,} paramètres)"
    )
    return (model_c,)


@app.cell
def _(torch):
    # --- Hyperparamètres + critère (LR plus bas car on part de poids appris) ---
    NUM_EPOCHS = 25
    LEARNING_RATE = 5e-4
    PATIENCE = 5

    criterion = torch.nn.CrossEntropyLoss()
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
        return {"history": history, "best_state": best_state, "best_val_loss": best_val_loss}

    return (train_model,)


@app.cell
def _(
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    criterion,
    mo,
    model_c,
    train_loader,
    train_model,
    val_loader,
):
    # === LANCER L'ENTRAÎNEMENT (cellule lourde — exécution manuelle) ===
    training_result_c = train_model(
        model_c,
        train_loader,
        val_loader,
        criterion,
        NUM_EPOCHS,
        LEARNING_RATE,
        PATIENCE,
    )
    model_c.load_state_dict(training_result_c["best_state"])
    mo.md(
        "## Entraînement terminé\n"
        f"- Meilleure val loss : **{training_result_c['best_val_loss']:.4f}**\n"
        f"- Epochs effectuées : {len(training_result_c['history']['train_loss'])}"
    )
    return (training_result_c,)


@app.cell
def _(mo, plt, training_result_c):
    # --- Courbes loss / accuracy ---
    history_c = training_result_c["history"]
    epochs_axis = range(1, len(history_c["train_loss"]) + 1)

    fig_curves, axes_curves = plt.subplots(1, 2, figsize=(12, 4))
    axes_curves[0].plot(epochs_axis, history_c["train_loss"], label="train")
    axes_curves[0].plot(epochs_axis, history_c["val_loss"], label="val")
    axes_curves[0].set_title("Loss")
    axes_curves[0].set_xlabel("epoch")
    axes_curves[0].legend()
    axes_curves[1].plot(epochs_axis, history_c["train_acc"], label="train")
    axes_curves[1].plot(epochs_axis, history_c["val_acc"], label="val")
    axes_curves[1].set_title("Accuracy")
    axes_curves[1].set_xlabel("epoch")
    axes_curves[1].legend()
    fig_curves.tight_layout()
    mo.vstack([mo.md("## Courbes d'apprentissage"), fig_curves])
    return


@app.cell
def _(CLASS_NAMES, Path, mo, torch, training_result_c):
    # --- Sauvegarde du modèle 3 classes ---
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_C_PATH = MODEL_DIR / "model_c_pneumonia_type.pt"

    torch.save(
        {
            "model_state": training_result_c["best_state"],
            "history": training_result_c["history"],
            "classes": CLASS_NAMES,
            "img_size": 224,
            "arch": "SimpleCNN-3classes",
            "from": "model_a_cnn.pt",
        },
        MODEL_C_PATH,
    )
    mo.md(f"## Modèle sauvegardé\n`{MODEL_C_PATH.resolve()}`")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Évaluation (3 classes)

    Rechargement depuis le disque puis métriques multi-classes :
    accuracy globale, rapport par classe (précision/rappel/F1) et matrice
    de confusion 3×3.
    """)
    return


@app.cell
def _(Path, device, mo, torch):
    # --- Rechargement du modèle 3 classes ---
    from model_a_cnn import SimpleCNN as SimpleCNN3

    EVAL_MODEL_PATH = Path("models/model_c_pneumonia_type.pt")
    assert EVAL_MODEL_PATH.exists(), (
        f"Modèle introuvable : {EVAL_MODEL_PATH}. Lance l'entraînement + sauvegarde."
    )

    eval_checkpoint = torch.load(
        EVAL_MODEL_PATH, map_location=device, weights_only=False
    )
    model_c_eval = SimpleCNN3(num_classes=2)
    model_c_eval.classifier[-1] = torch.nn.Linear(
        model_c_eval.classifier[-1].in_features, 3
    )
    model_c_eval.load_state_dict(eval_checkpoint["model_state"])
    model_c_eval = model_c_eval.to(device)
    model_c_eval.eval()
    mo.md(f"Modèle rechargé depuis `{EVAL_MODEL_PATH}` ({eval_checkpoint['arch']})")
    return (model_c_eval,)


@app.cell
def _(device, model_c_eval, test_loader, torch):
    # --- Prédictions sur le set de test ---
    def collect_predictions(model, loader):
        labels_all, preds_all = [], []
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images.to(device))
                labels_all.extend(labels.tolist())
                preds_all.extend(logits.argmax(1).cpu().tolist())
        return labels_all, preds_all

    y_true, y_pred = collect_predictions(model_c_eval, test_loader)
    return y_pred, y_true


@app.cell
def _(CLASS_NAMES, mo, pl, y_pred, y_true):
    # --- Métriques par classe ---
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    per_class_df = pl.DataFrame(
        {
            "classe": CLASS_NAMES,
            "précision": [round(report[c]["precision"], 4) for c in CLASS_NAMES],
            "rappel": [round(report[c]["recall"], 4) for c in CLASS_NAMES],
            "f1": [round(report[c]["f1-score"], 4) for c in CLASS_NAMES],
            "support": [int(report[c]["support"]) for c in CLASS_NAMES],
        }
    )
    global_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mo.vstack(
        [
            mo.md(
                "## Métriques (test)\n"
                f"- **Accuracy globale** : {global_acc:.4f}\n"
                f"- **F1 macro** : {macro_f1:.4f}"
            ),
            per_class_df,
        ]
    )
    return (classification_report,)


@app.cell
def _(CLASS_NAMES, classification_report, y_pred, y_true):
    cr = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(cr)
    return


@app.cell
def _(CLASS_NAMES, mo, plt, y_pred, y_true):
    # --- Matrice de confusion 3x3 ---
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 5))
    im_cm = ax_cm.imshow(conf_matrix, cmap="Purples")
    ax_cm.set_xticks(range(3))
    ax_cm.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax_cm.set_yticks(range(3))
    ax_cm.set_yticklabels(CLASS_NAMES)
    ax_cm.set_xlabel("Prédit")
    ax_cm.set_ylabel("Réel")
    ax_cm.set_title("Matrice de confusion (3 classes)")
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
def _():
    return


if __name__ == "__main__":
    app.run()
