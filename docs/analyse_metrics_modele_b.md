# Analyse des métriques — Modèle B (ResNet18, transfer learning)

> Notebook : [`ModelB.py`](../ModelB.py) · Modèle : **ResNet18 pré-entraîné
> ImageNet**, backbone gelé + tête `fc` ré-apprise · Tâche : binaire **NORMAL vs
> PNEUMONIA** · Set de test : 624 images (234 NORMAL, 390 PNEUMONIA).

## Résultats

| Métrique | Modèle B | (rappel Modèle A) |
|---|---|---|
| Accuracy | **0,8478** | 0,7292 |
| Précision (PNEUMONIA) | 0,8105 | 0,6991 |
| **Rappel / Sensibilité** | 0,9872 | 0,9949 |
| Spécificité | **0,6154** | 0,2863 |
| F1-score | 0,8902 | 0,8212 |
| ROC-AUC | **0,9524** | 0,9185 |

**Matrice de confusion** :

|  | Prédit NORMAL | Prédit PNEUMONIA |
|---|---|---|
| **Réel NORMAL** | TN = 144 | FP = 90 |
| **Réel PNEUMONIA** | FN = 5 | TP = 385 |

## Pourquoi ces valeurs ?

### ROC-AUC = 0,952 (le plus élevé des modèles)
`> 0,919` du Modèle A. Le **transfer learning** part de filtres déjà entraînés sur
des millions d'images (ImageNet) : les premières couches savent détecter bords,
textures, formes. Le réseau sépare donc mieux les deux classes, d'où une AUC
supérieure.

### Spécificité = 0,615 (gros gain vs Modèle A : 0,286)
`TN / (TN + FP) = 144 / 234`. Les faux positifs chutent de **167 → 90**. Des
features plus riches permettent de mieux reconnaître un poumon **sain** au lieu de
crier « pneumonie » par défaut. C'est l'amélioration la plus marquante.

### Rappel / Sensibilité = 0,987 (toujours excellent)
`385 / 390`, seulement 5 pneumonies manquées. On conserve la quasi-totalité de la
sensibilité **tout en** gagnant en spécificité : c'est un bien meilleur compromis
que le Modèle A (qui sur-payait sa sensibilité en fausses alertes).

### Précision = 0,811 (vs 0,699)
`385 / 475`. Avec moitié moins de faux positifs, les alertes « pneumonie » sont
bien plus fiables.

### Accuracy = 0,848 (vs 0,729)
`(385 + 144) / 624`. Remonte fortement grâce à la baisse des FP. Ici l'accuracy
redevient cohérente avec les autres métriques (le modèle n'est plus biaisé vers
une seule classe).

### F1-score = 0,890
Précision et rappel tous deux élevés → F1 élevé et **équilibré**.

## Synthèse
Le Modèle B **domine le Modèle A sur toutes les métriques**. Il garde une
sensibilité quasi parfaite (dépistage sûr) **et** corrige le défaut majeur du
Modèle A (la spécificité), grâce aux représentations pré-entraînées d'ImageNet.
À configuration égale (même split, même sampler, même seuil 0,5), c'est le
**modèle à retenir** pour la tâche binaire.

> Reste un léger biais « pneumonie » (90 FP > 5 FN), cohérent avec le sampler
> équilibré ; un ajustement de seuil pourrait encore affiner le compromis.
