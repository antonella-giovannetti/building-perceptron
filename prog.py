import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def load_data(file_path=None):

    if file_path:
        data = pd.read_csv(file_path)
        X = data.drop(columns=["label"]) 
        y = data["label"]
    else:#génération aléatoire des données
        X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2,
        n_informative=8, random_state=42
        )

# étiquettes
        y = pd.Series(y).apply(lambda val: 1 if val > 0 else -1)

#normalisation des caractéristiques
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

#données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test