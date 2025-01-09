
import matplotlib.pyplot as plt 
import seaborn as sns 
from perceptron_ import PerceptronModel 
from per_trait import load_data, clean_data, encode_target, preprocess_data, split_data 


#données
data = load_data("bcw_data.csv") 

#netoyage des données 
data_cleaned = clean_data(data) 


#distribution de la variable cible 
sns.countplot(data=data_cleaned, x='diagnosis', palette='Set2') 
plt.title('Distribution du diagnostic') 
plt.xlabel('Diagnostic (B = Benigne, M = Maligne)') 
plt.ylabel('Count') 
plt.show() 

#traitement des données 
data_cleaned = encode_target(data_cleaned) 
X_pca, y = preprocess_data(data_cleaned)


 #séparation des données en ensemble d'entraînement et de test 
X_train, X_test, y_train, y_test = split_data(X_pca, y) 


#création et entraînement du modèle Perceptron 
perceptron_model = PerceptronModel(max_iter=500) 
perceptron_model.fit(X_train, y_train) 

#prédictions 
y_pred = perceptron_model.predict(X_test)
 
# évaluation du modèle 
accuracy, report, cm = perceptron_model.evaluate(y_test, y_pred)
 # Affichage des résultats 
print(f"Accuracy: {accuracy}") 
print("\nClassification Report:") 
print(report) 
print("\nConfusion Matrix:") 
print(cm)