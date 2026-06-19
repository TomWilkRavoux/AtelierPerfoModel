# Métriques du Modèle A & pistes d'amélioration

> Lecture des résultats du **Modèle A** (CNN maison) sur le set de test, analyse
> du **pourquoi l'accuracy n'est qu'à 0,73**, et pistes concrètes pour
> l'améliorer. Code d'évaluation : Phase 5 de [`Exploration.py`](../Exploration.py).

---

## 1. Résultats obtenus (set de test, 624 images)

| Métrique | Valeur |
|---|---|
| Accuracy | **0,729** |
| Précision (PNEUMONIA) | 0,699 |
| **Rappel / Sensibilité** | **0,995** |
| Spécificité | 0,286 |
| F1-score | 0,821 |
| ROC-AUC | **0,919** |

**Matrice de confusion** (classe positive = PNEUMONIA) :

|  | Prédit NORMAL | Prédit PNEUMONIA |
|---|---|---|
| **Réel NORMAL** | TN = 67 | FP = 167 |
| **Réel PNEUMONIA** | FN = 2 | TP = 388 |

---

## 2. Comprendre chaque métrique

Avec :
- **TP** (vrais positifs) = pneumonies correctement détectées = 388
- **TN** (vrais négatifs) = sains correctement classés = 67
- **FP** (faux positifs) = sains classés « pneumonie » (fausse alerte) = 167
- **FN** (faux négatifs) = pneumonies ratées = 2

### Accuracy — `(TP + TN) / total`
Proportion de bonnes prédictions, toutes classes confondues.
`(388 + 67) / 624 = 0,729`. **Trompeuse sur données déséquilibrées** (voir §3).

### Précision — `TP / (TP + FP)`
Quand le modèle dit « pneumonie », à quel point a-t-il raison ?
`388 / (388 + 167) = 0,699` → ~30 % de ses alertes sont fausses.

### Rappel / Sensibilité — `TP / (TP + FN)`
Parmi toutes les vraies pneumonies, combien sont détectées ?
`388 / (388 + 2) = 0,995` → **quasi aucune pneumonie ratée**. En médecine,
c'est la métrique reine pour un dépistage.

### Spécificité — `TN / (TN + FP)`
Parmi tous les sains, combien sont correctement reconnus comme sains ?
`67 / (67 + 167) = 0,286` → **point faible** : le modèle « crie au loup » trop
souvent.

### F1-score — moyenne harmonique précision/rappel
`0,821` : bon, mais tiré vers le haut par l'excellent rappel et pénalisé par la
précision moyenne.

### ROC-AUC — `0,919`
Mesure la capacité à **séparer** les deux classes **indépendamment du seuil**.
0,5 = aléatoire, 1,0 = parfait. **0,92 est élevé** : l'information discriminante
est bien là — le problème vient surtout du **seuil de décision** (§4).

---

## 3. Pourquoi l'accuracy n'est qu'à 0,73 ?

Trois causes se combinent.

### 3.1 Un seuil de décision biaisé (cause principale)
On a entraîné avec un **`WeightedRandomSampler`** qui rééquilibre les classes
(le train réel est ~3:1 en faveur de PNEUMONIA). Le modèle a donc appris sur une
distribution **50/50**, mais le **test reste déséquilibré** et l'on décide avec
un **seuil fixe de 0,5**. Résultat : le modèle a tendance à prédire « pneumonie »
au moindre doute → **167 faux positifs**. C'est ce qui écrase la spécificité (et
donc l'accuracy), alors que l'AUC de 0,92 prouve que le modèle *sait* distinguer
les classes.

### 3.2 L'accuracy est trompeuse ici
Le test contient 390 pneumonies pour 234 sains (62 % / 38 %). Un modèle qui
prédirait *toujours* « pneumonie » ferait déjà **62 % d'accuracy** sans rien
apprendre. L'accuracy seule n'est donc pas un bon juge → on regarde rappel,
spécificité et AUC.

### 3.3 Un modèle volontairement simple
Le Modèle A est une **baseline légère** (~0,5 M paramètres, entraînée de zéro)
sur ~5 000 images. Sa capacité de généralisation est limitée comparée à un réseau
profond pré-entraîné sur des millions d'images.

---

## 4. Pistes d'amélioration

### 4.1 Ajuster le seuil de décision *(gain rapide, sans ré-entraîner)*
Plutôt que de trancher à 0,5 sur la probabilité de pneumonie, **choisir le seuil**
à partir de la courbe ROC / précision-rappel (p. ex. indice de Youden, ou un seuil
qui garantit une sensibilité ≥ 0,95 tout en maximisant la spécificité). Avec une
AUC de 0,92, on peut **remonter fortement la spécificité** en ne sacrifiant qu'un
peu de rappel. C'est l'action la plus rentable immédiatement.

### 4.2 Améliorer le CNN lui-même (architecture & filtres)
- **Réseau plus profond / plus de filtres** : augmenter le nombre de canaux par
  bloc (32→64→128→256→512) et ajouter des blocs pour capter des motifs plus fins.
- **Connexions résiduelles (ResNet-like)** : faciliter l'entraînement des
  réseaux profonds et améliorer la généralisation.
- **Prétraitement / filtres image** : les radios se prêtent bien à des filtres
  classiques en amont — **CLAHE** (égalisation d'histogramme adaptative pour
  renforcer le contraste des opacités), débruitage gaussien, ou détection de
  contours (Sobel) en canal additionnel. Cela peut aider un petit CNN à mieux
  « voir » les zones pathologiques.
- **Filtres plus grands au 1er étage** (ex. 5×5 ou 7×7) pour élargir le champ
  réceptif initial, comme dans les architectures éprouvées.

### 4.3 Transfer learning *(Modèle B — le plus prometteur)*
Repartir d'un réseau **pré-entraîné sur ImageNet** (ResNet18, DenseNet121,
EfficientNet) et le *fine-tuner*. Sur ce dataset, c'est généralement ce qui donne
le meilleur compromis sensibilité/spécificité, car les premières couches
détectent déjà des textures utiles. → traité dans un notebook séparé.

### 4.4 Mieux gérer le déséquilibre
- Tester une **loss pondérée** (`CrossEntropyLoss(weight=class_weights)`) ou la
  **Focal Loss** au lieu (ou en complément raisonné) du sampler.
- Comparer « sampler » vs « loss pondérée » : le sampler peut sur-corriger et
  c'est en partie ce qui biaise le seuil (§3.1).

### 4.5 Augmentation de données plus riche
Ajouter zoom aléatoire, translations, variations de luminosité/contraste,
*random erasing*. Cela régularise et améliore la robustesse sur le test.

### 4.6 Régularisation & entraînement
- **Early stopping sur une métrique métier** (ex. F1 ou recall) plutôt que sur la
  val loss.
- Ajuster `dropout`, *weight decay*, et tester un **learning rate scheduling**
  plus fin (warmup, cosine).
- Entraîner plus longtemps avec un meilleur matériel (GPU) pour explorer des
  modèles plus gros.

### 4.7 Résolution d'entrée
Passer de 224×224 à une résolution plus élevée (ex. 320×320) peut révéler des
détails fins, au prix d'un coût de calcul accru.

---

## 5. Synthèse

Le Modèle A est une **bonne baseline de dépistage** : il ne rate presque aucune
pneumonie (sensibilité 0,995) et sépare bien les classes (AUC 0,92). Son
**accuracy faible (0,73) n'est pas un défaut de “savoir”, mais de “seuil”** : il
sur-déclenche, d'où 167 fausses alertes.

**Ordre de priorité conseillé** :
1. Ajuster le **seuil de décision** (gain immédiat sur la spécificité) ;
2. Passer au **transfer learning** (Modèle B) ;
3. Affiner l'**architecture / les filtres** et la **gestion du déséquilibre**.

Ces leviers serviront aussi de **justification** dans la présentation (Phase 6)
et de grille de comparaison avec le Modèle B.
