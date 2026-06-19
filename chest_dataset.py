"""Dataset PyTorch pour les radiographies thoraciques.

Défini dans un module séparé (et non dans une cellule marimo) pour qu'il soit
*picklable* : les workers de `DataLoader` (num_workers > 0, lancés en `spawn`)
doivent pouvoir importer cette classe.
"""

from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    """Charge une radio (chemin, label), la convertit en RGB 3 canaux (requis
    par les modèles pré-entraînés ImageNet) puis applique le transform."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
