import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split 
from boruta import BorutaPy 
from sklearn.ensemble import RandomForestClassifier 


def load_data(filepath): # Chargement des données depuis un fichier CSV 
    data = pd.read_csv(filepath) 
    return data 

def clean_data(data): # Suppression des colonnes inutiles 
    data_cleaned = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore') 
    return data_cleaned 

def encode_target(data_cleaned): # Encodage de la variable cible (B -> 0, M -> 1) 
    data_cleaned['diagnosis'] = data_cleaned['diagnosis'].map({'B': 0, 'M': 1}) 
    return data_cleaned 


def preprocess_data(data_cleaned): # Séparation des caractéristiques (X) et de la cible (y) 
    X = data_cleaned.drop(columns=['diagnosis']) 
    y = data_cleaned['diagnosis'] # Normalisation des données 
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X)
    
 # Sélection des caractéristiques avec Boruta 
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5) 
    boruta = BorutaPy(rf, n_estimators='auto', random_state=42) 
    boruta.fit(X_scaled, y) 
    
# Récupérer les caractéristiques sélectionnées par Boruta 
    selected_features = X.columns[boruta.support_].to_list() 
    print(f"Caractéristiques sélectionnées par Boruta : {selected_features}") 
    
    
# Réduire X aux caractéristiques sélectionnées 
    X_selected = X_scaled[:, boruta.support_]
    
    
#ACP
    pca = PCA(n_components=10) 
    X_pca = pca.fit_transform(X_selected) 
    return X_pca, y 


def split_data(X_pca, y, test_size=0.2, random_state=42): 
# Séparation des données en ensemble d'entraînement et de test 
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=random_state) 
    return X_train, X_test, y_train, y_test