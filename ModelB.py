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
    # Modèle B — Transfer learning (ResNet18 pré-entraîné)

    Notebook **autonome** (kernel séparé) qui reprend les phases du plan pour
    un second modèle, par **transfer learning** :
    - prépa data identique au Modèle A (re-split, transforms, sampler) ;
    - **ResNet18 pré-entraîné sur ImageNet**, dont on remplace la tête ;
    - entraînement + sauvegarde ;
    - évaluation sur le test ;
    - **comparaison Modèle A vs Modèle B**.

    > Par défaut on fait de l'**extraction de caractéristiques** (backbone
    > gelé, on n'entraîne que la nouvelle tête) → rapide et efficace sur CPU.
    > Passer `FREEZE_BACKBONE = False` pour un fine-tuning complet (plus long).
    """)
    return


@app.cell
def _(mo, torch):
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo.md(f"**Device : `{device}`**")
    return (device,)


@app.cell
def _(Path, mo):
    # --- Données ---
    DATA_DIR = Path("asset/chest_xray")
    CLASSES = ["NORMAL", "PNEUMONIA"]
    IMG_EXTS = {".jpeg", ".jpg", ".png"}
    LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    assert DATA_DIR.exists(), f"Dossier introuvable : {DATA_DIR.resolve()}"
    mo.md(f"Dataset : `{DATA_DIR.resolve()}`")
    return CLASSES, DATA_DIR, IMG_EXTS, LABEL_MAP


@app.cell
def _(CLASSES, DATA_DIR, IMG_EXTS, LABEL_MAP, mo, pl):
    # --- Re-split stratifié (identique au Modèle A pour une comparaison juste) ---
    from sklearn.model_selection import train_test_split

    def list_images(split, cls):
        folder = DATA_DIR / split / cls
        return (
            [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
            if folder.exists()
            else []
        )

    combined_samples = [
        (path, LABEL_MAP[cls])
        for split in ("train", "val")
        for cls in CLASSES
        for path in list_images(split, cls)
    ]
    test_samples = [
        (path, LABEL_MAP[cls])
        for cls in CLASSES
        for path in list_images("test", cls)
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
            "total": [len(train_samples), len(val_samples), len(test_samples)],
        }
    )
    mo.vstack([mo.md("## Re-split (85/15)"), split_summary])
    return test_samples, train_samples, val_samples


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
    # --- Datasets + DataLoaders avec WeightedRandomSampler ---
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from chest_dataset import ChestXrayDataset

    train_ds = ChestXrayDataset(train_samples, train_transform)
    val_ds = ChestXrayDataset(val_samples, eval_transform)
    test_ds = ChestXrayDataset(test_samples, eval_transform)

    # Poids pour rééquilibrer les classes au tirage
    train_class_counts = [
        sum(1 for _, label in train_samples if label == 0),
        sum(1 for _, label in train_samples if label == 1),
    ]
    class_weights = [
        len(train_samples) / (2 * count) for count in train_class_counts
    ]
    sample_weights = [class_weights[label] for _, label in train_samples]
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
        f"DataLoaders prêts (batch={BATCH_SIZE}, workers={NUM_WORKERS}) — "
        f"train {len(train_loader)} / val {len(val_loader)} / test {len(test_loader)} batches"
    )
    return test_loader, train_loader, val_loader


@app.cell
def _(device, mo, torch):
    # --- Modèle B : ResNet18 pré-entraîné, tête remplacée ---
    from torchvision import models

    FREEZE_BACKBONE = True  # True = extraction de features (rapide) ; False = fine-tuning complet

    def build_model_b(num_classes=2, freeze=True):
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
        if freeze:
            for param in net.parameters():
                param.requires_grad = False
        # Nouvelle tête de classification (toujours entraînable)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        return net

    model_b = build_model_b(num_classes=2, freeze=FREEZE_BACKBONE).to(device)

    trainable_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    total_b = sum(p.numel() for p in model_b.parameters())
    mo.md(
        "## Modèle B instancié\n"
        f"- Backbone gelé : **{FREEZE_BACKBONE}**\n"
        f"- Paramètres entraînables : **{trainable_b:,}** / {total_b:,} total"
    )
    return FREEZE_BACKBONE, model_b


@app.cell
def _(torch):
    # --- Hyperparamètres + critère ---
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-3
    PATIENCE = 4

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
        # On n'optimise que les paramètres entraînables (tête si backbone gelé)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
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
    model_b,
    train_loader,
    train_model,
    val_loader,
):
    # === LANCER L'ENTRAÎNEMENT (cellule lourde — exécution manuelle) ===
    training_result_b = train_model(
        model_b,
        train_loader,
        val_loader,
        criterion,
        NUM_EPOCHS,
        LEARNING_RATE,
        PATIENCE,
    )
    model_b.load_state_dict(training_result_b["best_state"])
    mo.md(
        "## Entraînement terminé\n"
        f"- Meilleure val loss : **{training_result_b['best_val_loss']:.4f}**\n"
        f"- Epochs effectuées : {len(training_result_b['history']['train_loss'])}"
    )
    return (training_result_b,)


@app.cell
def _(mo, plt, training_result_b):
    # --- Courbes loss / accuracy ---
    history_b = training_result_b["history"]
    epochs_axis = range(1, len(history_b["train_loss"]) + 1)

    fig_curves, axes_curves = plt.subplots(1, 2, figsize=(12, 4))
    axes_curves[0].plot(epochs_axis, history_b["train_loss"], label="train")
    axes_curves[0].plot(epochs_axis, history_b["val_loss"], label="val")
    axes_curves[0].set_title("Loss")
    axes_curves[0].set_xlabel("epoch")
    axes_curves[0].legend()
    axes_curves[1].plot(epochs_axis, history_b["train_acc"], label="train")
    axes_curves[1].plot(epochs_axis, history_b["val_acc"], label="val")
    axes_curves[1].set_title("Accuracy")
    axes_curves[1].set_xlabel("epoch")
    axes_curves[1].legend()
    fig_curves.tight_layout()
    mo.vstack([mo.md("## Courbes d'apprentissage"), fig_curves])
    return


@app.cell
def _(FREEZE_BACKBONE, Path, mo, torch, training_result_b):
    # --- Sauvegarde du modèle entraîné ---
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_B_PATH = MODEL_DIR / "model_b_resnet18.pt"

    torch.save(
        {
            "model_state": training_result_b["best_state"],
            "history": training_result_b["history"],
            "classes": ["NORMAL", "PNEUMONIA"],
            "img_size": 224,
            "arch": "resnet18",
            "freeze_backbone": FREEZE_BACKBONE,
        },
        MODEL_B_PATH,
    )
    mo.md(f"## Modèle sauvegardé\n`{MODEL_B_PATH.resolve()}`")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Évaluation du Modèle B

    Rechargement depuis le disque (indépendant de l'entraînement), puis
    mêmes métriques que pour le Modèle A.
    """)
    return


@app.cell
def _(Path, device, mo, torch):
    # --- Rechargement du modèle entraîné ---
    from torchvision import models as models_eval

    EVAL_MODEL_PATH = Path("models/model_b_resnet18.pt")
    assert EVAL_MODEL_PATH.exists(), (
        f"Modèle introuvable : {EVAL_MODEL_PATH}. Lance l'entraînement + sauvegarde."
    )

    eval_checkpoint = torch.load(
        EVAL_MODEL_PATH, map_location=device, weights_only=False
    )
    model_b_eval = models_eval.resnet18(weights=None)
    model_b_eval.fc = torch.nn.Linear(model_b_eval.fc.in_features, 2)
    model_b_eval.load_state_dict(eval_checkpoint["model_state"])
    model_b_eval = model_b_eval.to(device)
    model_b_eval.eval()
    mo.md(
        "## Modèle rechargé\n"
        f"- Source : `{EVAL_MODEL_PATH}`\n"
        f"- Architecture : `{eval_checkpoint['arch']}` "
        f"(backbone gelé : {eval_checkpoint['freeze_backbone']})"
    )
    return (model_b_eval,)


@app.cell
def _(device, model_b_eval, test_loader, torch):
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

    y_true, y_pred, y_prob = collect_predictions(model_b_eval, test_loader)
    return y_pred, y_prob, y_true


@app.cell
def _(CLASSES, mo, pl, y_pred, y_prob, y_true):
    # --- Métriques ---
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
    tn, fp, fn, tp = conf_matrix.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    spec = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cr = classification_report(y_true, y_pred, target_names=CLASSES)

    metrics_b = {
        "Accuracy": round(float(acc), 4),
        "Précision (PNEUMONIA)": round(float(prec), 4),
        "Rappel / Sensibilité": round(float(rec), 4),
        "Spécificité": round(float(spec), 4),
        "F1-score": round(float(f1), 4),
        "ROC-AUC": round(float(auc), 4),
    }
    metrics_df_b = pl.DataFrame(
        {"métrique": list(metrics_b.keys()), "valeur": list(metrics_b.values())}
    )

    print(cr)

    mo.vstack([mo.md("## Métriques Modèle B (test)"), metrics_df_b])
    return auc, conf_matrix, metrics_b


@app.cell
def _(CLASSES, conf_matrix, mo, plt):
    # --- Matrice de confusion ---
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    im_cm = ax_cm.imshow(conf_matrix, cmap="Greens")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_xticklabels(CLASSES)
    ax_cm.set_yticks([0, 1])
    ax_cm.set_yticklabels(CLASSES)
    ax_cm.set_xlabel("Prédit")
    ax_cm.set_ylabel("Réel")
    ax_cm.set_title("Matrice de confusion — Modèle B (test)")
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
    ax_roc.set_title("Courbe ROC — Modèle B (test)")
    ax_roc.legend()
    fig_roc.tight_layout()
    mo.vstack([mo.md("## Courbe ROC"), fig_roc])
    return


@app.cell
def _(metrics_b, mo, pl):
    # --- Comparaison Modèle A vs Modèle B ---
    # Métriques du Modèle A : issues de Exploration.py (Phase 5, set de test).
    metrics_a = {
        "Accuracy": 0.7292,
        "Précision (PNEUMONIA)": 0.6991,
        "Rappel / Sensibilité": 0.9949,
        "Spécificité": 0.2863,
        "F1-score": 0.8212,
        "ROC-AUC": 0.9185,
    }
    comparison_df = pl.DataFrame(
        {
            "métrique": list(metrics_a.keys()),
            "Modèle A (CNN maison)": list(metrics_a.values()),
            "Modèle B (ResNet18)": [metrics_b[k] for k in metrics_a],
        }
    )
    mo.vstack(
        [
            mo.md(
                "## Comparaison Modèle A vs Modèle B\n"
                "Le modèle retenu sera celui qui offre le meilleur compromis "
                "**sensibilité / spécificité** (et l'AUC la plus élevée)."
            ),
            comparison_df,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
