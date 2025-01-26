# building-perceptron
# **Classification des données avec Perceptron**

## **Contexte du Projet**
Ce projet explore la classification binaire des tumeurs (bénignes ou malignes) en utilisant un modèle de perceptron implémenté à l'aide de la bibliothèque scikit-learn. Le pipeline complet comprend le prétraitement des données, la réduction de la dimensionnalité et l'évaluation des performances du modèle.

---

## **Détails des étapes effectuées**

### **1. Préparation des Données**
Le jeu de données initial contient des caractéristiques extraites de cellules médicales, avec une variable cible indiquant si une tumeur est bénigne ("B") ou maligne ("M"). Les étapes suivantes ont été suivies :
- **Chargement des Données** : Lecture des données à partir d'un fichier CSV.
- **Nettoyage des Données** :
  - Suppression des valeurs manquantes pour éviter les biais.
  - Conversion de la variable cible en format numérique ("B" = 0, "M" = 1).
- **Normalisation** : Application d'une normalisation min-max pour homogéniser les échelles des variables et améliorer les performances du modèle.

### **2. Analyse Exploratoire des Données**
L'analyse exploratoire a permis de mieux comprendre la structure des données :
- **Statistiques Descriptives** : Analyse des distributions et des valeurs aberrantes.
- **Corrélations** : Étude des relations entre les variables pour identifier celles qui influencent le plus la cible.
- **Visualisations** :
  - Histogrammes et boxplots pour analyser la distribution des variables.
  - Matrices de corrélation pour observer les liens entre les caractéristiques.

### **3. Réduction de la Dimensionnalité**
Pour simplifier la complexité des données et améliorer les performances du modèle :
- **Analyse en Composantes Principales (ACP)** :
  - Sélection des 10 premières composantes, conservant 95 % de la variance totale.
  - Projection des données dans cet espace réduit pour réduire le bruit et les redondances.

### **4. Modélisation avec le Perceptron**
Un perceptron a été utilisé pour classer les tumeurs comme bénignes ou malignes :
- **Entraînement** : Le modèle a été ajusté sur les données transformées après ACP.
- **Prédictions** : Les étiquettes des échantillons de test ont été prédites.
- **Évaluation** :
  - **Accuracy** : Mesure du taux de prédictions correctes.
  - **Rapport de Classification** :
    - **Précision** : Taux de prédictions positives correctes.
    - **Rappel** : Taux de détection des instances positives.
    - **F1-Score** : Moyenne harmonique de la précision et du rappel.
  - **Matrice de Confusion** : Analyse des erreurs de classification (faux positifs et faux négatifs).

### **5. Outils et Bibliothèques Utilisés**
- **Pandas** : Gestion des données tabulaires.
- **NumPy** : Calculs numériques avancés.
- **Matplotlib/Seaborn** : Visualisation graphique.
- **Scikit-learn** :
  - ACP : Réduction de dimensionnalité.
  - Perceptron : Implémentation de l'algorithme.
  - Métriques : Calcul de l'accuracy, du rapport de classification et de la matrice de confusion.

### **6. Résultats Obtenus**
Le modèle a atteint une précision moyenne de **95,6 %**, avec une bonne distinction entre les classes malignes et bénignes. La matrice de confusion montre un faible taux d'erreurs, tandis que les scores F1 confirment une bonne équilibre entre précision et rappel.

---

## **Conclusion**
Ce projet illustre comment un perceptron peut être utilisé pour une classification efficace lorsqu'il est combiné avec des techniques appropriées de prétraitement et de réduction de dimensionnalité. Bien que les perceptrons soient des modèles simples, ils peuvent fournir des performances impressionnantes dans des scénarios bien structurés. Une exploration future pourrait inclure l'utilisation de modèles non linéaires ou de réseaux de neurones pour améliorer davantage les résultats.

---

