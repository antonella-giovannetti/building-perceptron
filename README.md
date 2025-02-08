# Projet : Implémentation d'un Perceptron personnalisé pour la classification

## Contexte du projet
Ce projet a pour objectif de construire un perceptron à partir de zéro, sans utiliser de bibliothèques comme scikit-learn pour l'implémentation du modèle. Nous allons utiliser les concepts fondamentaux de l'apprentissage automatique pour créer, entraîner et évaluer un perceptron sur un jeu de données de classification.

Les étapes principales incluent :
- Préparation et nettoyage des données.
- Implémentation d'un perceptron personnalisé.
- Entraînement et évaluation des performances du modèle.
- Documentation et analyse des résultats.

## Données utilisées
Les données utilisées proviennent d'un fichier CSV nommé `bcw_data.csv`. Il s'agit de données sur le cancer du sein qui comprennent :
- **Attributs** : Caractéristiques des cellules tumorales.
- **Label cible** : Malin (1) ou Bénin (0).

Les données ont été nettoyées pour traiter les valeurs manquantes et normalisées pour prévenir les problèmes de mise à l'échelle.

## Analyse exploratoire
Avant d'entraîner le modèle, une analyse exploratoire a été réalisée pour comprendre la distribution des données et leur variabilité. Une réduction de dimensionnalité via l'Analyse en Composantes Principales (ACP) a été effectuée pour simplifier les données à 10 composantes principales tout en conservant l'essentiel de l'information.

## Implémentation du Perceptron
### Description
Un perceptron est un algorithme d'apprentissage supervisé qui effectue une classification binaire en utilisant une règle de mise à jour des poids. L'algorithme ajuste les poids en fonction des erreurs commises lors de la prédiction.

### Code
La classe `CustomPerceptron` implémente :
- **Initialisation** : Paramètres comme le taux d'apprentissage et le nombre d'itérations.
- **Entraînement** : Mise à jour des poids et du biais pour minimiser les erreurs.
- **Prédiction** : Classification des échantillons de test.
- **Évaluation** : Calcul de l'accuracy, du rapport de classification et de la matrice de confusion.

## Scripts

### 1. Traitement des données (`perceptron-analysis.ipynb`)
Ce script contient les fonctions pour :
- Charger les données depuis le fichier CSV.
- Nettoyer et normaliser les données.
- Appliquer l'ACP pour réduire la dimensionnalité.

### 2. Perceptron personnalisé (`perceptron.py`)
Implémentation d'un perceptron maison avec les méthodes `fit`, `predict` et `evaluate`.

### 3. Script principal (`main.py`)
- Charge les données préparées.
- Entraîne le perceptron.
- Prédit les résultats sur le jeu de test.
- Évalue les performances du modèle.

## Résultats
Les performances obtenues incluent :
- **Accuracy** : Indique la proportion de bonnes classifications.
- **Matrice de confusion** : Montre les vrais positifs, vrais négatifs, faux positifs et faux négatifs.
- **Rapport de classification** : Précision, rappel et F1-score pour chaque classe.

Exemple de résultat :
```
Accuracy: 0.9824561403508771

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99        71
           1       0.98      0.98      0.98        43

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114


Confusion Matrix:
 [[70  1]
 [ 1 42]]
```

## Conclusion
Ce projet démontre comment implémenter un perceptron personnalisé et évaluer ses performances sur un jeu de données réel. Les résultats montrent que le perceptron est efficace pour ce type de classification binaire. Bien que les perceptrons soient des modèles simples, ils peuvent fournir des performances impressionnantes dans des scénarios bien structurés. Une exploration future pourrait inclure l'utilisation de modèles non linéaires ou de réseaux de neurones pour améliorer davantage les résultats.

## Bibliographie
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain.
- Scikit-learn documentation : https://scikit-learn.org
- NumPy documentation : https://numpy.org
