# Analyse des métriques — Modèle C (3 classes : sain / bactérien / viral)

> Notebook : [`PneumoniaType.py`](../PneumoniaType.py) · Modèle : **Modèle A
> rechargé**, tête remplacée par `Linear(128 → 3)` et fine-tuné · Tâche :
> **NORMAL / BACTÉRIENNE / VIRALE** · Set de test : 624 images
> (234 NORMAL, 242 bactériennes, 148 virales).

## Résultats globaux

| Métrique | Valeur |
|---|---|
| Accuracy | 0,6779 |
| **F1 macro** | 0,6473 |

## Résultats par classe

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| NORMAL | **0,9733** | **0,3120** | 0,4725 | 234 |
| BACTÉRIENNE | 0,8371 | 0,9132 | 0,8735 | 242 |
| VIRALE | 0,4526 | 0,8716 | 0,5958 | 148 |

**Matrice de confusion** (lignes = réel, colonnes = prédit) :

|  | Préd. NORMAL | Préd. BACT. | Préd. VIRALE |
|---|---|---|---|
| **Réel NORMAL** | 73 | 25 | **136** |
| **Réel BACT.** | 1 | 221 | 20 |
| **Réel VIRALE** | 1 | 18 | 129 |

## Pourquoi ces valeurs ?

### Accuracy 0,678 / F1 macro 0,647 — en net retrait
Passer de 2 à 3 classes rend la tâche bien plus dure : on ajoute la distinction
**bactérien vs viral**, peu marquée sur une radio. La littérature plafonne
souvent vers 60–75 % sur ce sous-problème → ces valeurs sont **dans l'ordre de
grandeur attendu**, pas une anomalie.

### Le problème principal : NORMAL très mal rappelé (recall 0,312)
La ligne NORMAL de la matrice est éloquente : sur 234 sains, **136 sont prédits
VIRALE** et seulement 73 NORMAL. Pourtant la **précision NORMAL est de 0,97** :
quand le modèle dit « NORMAL », il a presque toujours raison, mais il l'ose
rarement.

**Causes combinées** :
1. **Héritage du biais du Modèle A** : le Modèle A sur-prédisait déjà
   « pneumonie » (spécificité 0,29). En repartant de ses poids, le Modèle C
   garde une réticence à prédire « sain ».
2. **Sampler équilibré sur 3 classes** : VIRALE (classe minoritaire, 1345 en
   train) est fortement sur-échantillonnée. Le modèle apprend à prédire VIRALE
   « par défaut » sur les cas incertains, et les poumons sains à l'aspect diffus
   basculent vers VIRALE.
3. **Frontière NORMAL/VIRAL floue** : la pneumonie virale donne des opacités
   diffuses, parfois subtiles, visuellement proches d'un poumon peu contrasté.

### VIRALE : rappel 0,872 mais précision 0,453
Le modèle « attrape » bien les vraies virales (rappel élevé) **mais** sa colonne
VIRALE est polluée par 136 NORMAL + 20 bactériennes → précision faible. C'est la
classe « fourre-tout » vers laquelle partent les cas douteux.

### BACTÉRIENNE : la classe la mieux reconnue (F1 0,874)
Précision 0,84 et rappel 0,91. C'est la classe **majoritaire** en entraînement et
ses signes radiologiques (opacités focales/lobaires denses) sont plus
caractéristiques → le modèle la distingue nettement mieux.

### Pourquoi regarder le F1 macro et pas l'accuracy ?
Le **F1 macro** (0,647) moyenne les F1 par classe **à poids égal**, sans favoriser
les classes nombreuses. Il révèle le déséquilibre de performance entre classes
(NORMAL faible, BACT. forte) que l'accuracy globale masque.

## Synthèse
Le modèle gère bien **BACTÉRIENNE**, attrape les **VIRALES** au prix de fausses
alertes, et **sous-détecte NORMAL** (biais hérité + sur-échantillonnage VIRALE).
La distinction viral/bactérien reste partielle — limite **attendue** de la radio
seule.

**Pistes** : repartir du Modèle B (ResNet18, meilleures features) plutôt que du
Modèle A ; tester une **loss pondérée** plutôt que le sampler (réduit le biais
« VIRALE par défaut ») ; et garder NORMAL/PNEUMONIA en binaire si la priorité est
le dépistage, le type n'étant qu'une indication secondaire.
