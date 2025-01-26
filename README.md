# building-perceptron
# 1- Définition du machine learning et du Deep learning
Le Machine learning est un sous ensemble de l'intelligence artificielle qui permet aux machines d'apprendre et à s'améliorer.
Le deep learning est quand à lui un sous ensemble du machine learning qui utilise des réseaux de neurones artificiels.

La différence entre les deux est que les modèles de machine learning peuvent s'exécuter sans complexité avec une seule 
instance alors que le deep learning necéssite souvent des instanceq plus importantes.

Le Machine Learning résout les problèmes grâce aux statistiques et aux mathématiques. 
Le deep learning combine les statistiques et les mathématiques avec l'architecture des réseaux neuronaux.


# 2- Différents types  d'applications deep learning

 - Le computer vision (reconnaissance faciale, imagerie médicale)

 - prédiction de diagnostic médicaux

 - Voitures autonomes

 #
 - Le perceptron est un neurone artificiel, c'est une unité de réseau de neurone.
 - Les neurone biologiques recoivent des signaux au niveau de dendrites, venant des axones d'autres neurones, alors que le perceptron
  n'est que sa simplification mathématique et ce signaux sont représentés par des valeurs mathémtiques.

  - Sa fonction mahématique est y = ∑wixi+b, où xi sont les entrées (les caractéristiques traitées par le perceptron), wi les poids associés (qui détermine l'importance de chaque xi)
  et b le biais qui permet de déplacer la frontière de décision.
  

  - Une des règles d'apprentissage du perceptron est qu'une mise à jour des 
  poids et necessaire lorsqu'une erreur est commise lors de prise de décision.
  
  - La fonction d'actiation généralement utilisé par le perceptron es la foncton sigmoide
  a= 1/1+exp(-z)

  - Processus d'entrainement du perceptron 
  * initialisation des paramètres
  Initialisation des wi b avec des valeurs aléatoires.
  
  * ensemble d'entrainement 
  On dispose d'une ensemble d'entrainement binaire 
  
  *Calcul de la sortie du perceptron 

  *Calcul de l'erreur

  *Calcul du poids du  biais

  *Interpretation et convergence

  *Critère de convergence et limitation


  - Le perceptron est incapble de résoudre ds problèmes non linéaires.