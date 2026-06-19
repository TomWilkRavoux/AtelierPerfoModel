"""Architecture du Modèle A (CNN maison).

Sortie dans un module importable pour pouvoir recharger exactement la même
architecture dans plusieurs notebooks (Exploration.py l'utilise inline ; les
notebooks de fine-tuning l'importent ici pour garantir la correspondance des
poids sauvegardés).
"""

from torch import nn


class SimpleCNN(nn.Module):
    """CNN de base : entrée (B, 3, 224, 224) -> sortie (B, num_classes) logits."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),     # 224 -> 112
            self._conv_block(32, 64),    # 112 -> 56
            self._conv_block(64, 128),   # 56 -> 28
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
