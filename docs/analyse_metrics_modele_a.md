# Analyse des métriques — Modèle A (CNN maison)

> Notebook : [`Exploration.py`](../Exploration.py) · Modèle : `SimpleCNN`
> (entraîné de zéro) · Tâche : binaire **NORMAL vs PNEUMONIA** · Set de test :
> 624 images (234 NORMAL, 390 PNEUMONIA).
>
> Voir aussi : [pistes d'amélioration](metriques_et_pistes.md).

## Résultats

| Métrique | Valeur |
|---|---|
| Accuracy | 0,7292 |
| Précision (PNEUMONIA) | 0,6991 |
| **Rappel / Sensibilité** | **0,9949** |
| Spécificité | 0,2863 |
| F1-score | 0,8212 |
| ROC-AUC | 0,9185 |

**Matrice de confusion** :

|  | Prédit NORMAL | Prédit PNEUMONIA |
|---|---|---|
| **Réel NORMAL** | TN = 67 | FP = 167 |
| **Réel PNEUMONIA** | FN = 2 | TP = 388 |

## Pourquoi ces valeurs ?

### Rappel / Sensibilité = 0,995 (très élevé)
`TP / (TP + FN) = 388 / 390`. Le modèle ne rate quasiment **aucune pneumonie**
(2 faux négatifs). C'est la conséquence directe d'un modèle qui **prédit
« pneumonie » très facilement** : forcément, il attrape presque tous les vrais
positifs.

### Spécificité = 0,286 (très faible)
`TN / (TN + FP) = 67 / 234`. Revers de la médaille : sur 234 poumons sains, 167
sont classés à tort « pneumonie ». Le modèle **sur-déclenche**.

**Cause** : l'entraînement utilise un `WeightedRandomSampler` qui rééquilibre les
classes (le train réel est ~3:1 en faveur de PNEUMONIA). Le modèle apprend donc
sur une distribution 50/50 et, avec un **seuil de décision fixé à 0,5**, il
penche vers la classe « pneumonie » au moindre doute.

### Précision = 0,699
`TP / (TP + FP) = 388 / 555`. ~30 % des alertes « pneumonie » sont fausses :
conséquence mécanique des 167 faux positifs.

### Accuracy = 0,729 (trompeuse)
`(TP + TN) / 624`. Tirée vers le bas par les 167 FP. Et **peu informative ici** :
le test contient 62 % de pneumonies, donc un modèle qui dirait *toujours*
« pneumonie » ferait déjà 62 %. L'accuracy seule ne doit pas servir de juge.

### F1-score = 0,821
Moyenne harmonique précision/rappel : bon, mais porté par l'excellent rappel et
limité par la précision moyenne.

### ROC-AUC = 0,919 (élevé) — la métrique clé
Mesure la séparabilité des classes **indépendamment du seuil**. 0,92 signifie que
le modèle **sait** distinguer sain/malade : l'information est là. Le problème
n'est donc pas la « connaissance » du modèle mais le **seuil de décision** mal
placé (§ spécificité).

## Synthèse
Bonne **baseline de dépistage** : sensibilité quasi parfaite, AUC élevé. Faiblesse
nette en spécificité (sur-déclenchement dû au sampler + seuil 0,5). Levier le plus
rentable : **ajuster le seuil de décision** (cf. [pistes](metriques_et_pistes.md)).
Sert de point de comparaison au Modèle B.
