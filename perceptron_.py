from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class PerceptronModel:  # Initialisation du modèle Perceptron 
 def __init__(self, max_iter=500, random_state=42):
	 self.model = Perceptron(max_iter=max_iter, random_state=random_state) 
     
   #entrainement du modèle  
 def fit(self, X_train, y_train): 

	 self.model.fit(X_train, y_train)
     
    #prédiction sur les données 
 def predict(self, X_test): 

	 self.model.predict(X_test)
   #évaluation  
 def evaluate(self, y_test, y_pred): 
 
     accuracy = accuracy_score(y_test, y_pred) 
     report = classification_report(y_test, y_pred) 
     cm = confusion_matrix(y_test, y_pred) 
     return accuracy, report, cm