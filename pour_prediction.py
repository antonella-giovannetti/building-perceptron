import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("bcw_data.csv")
if 'Unnamed: 32' in data.columns:
    data.drop(columns=['Unnamed: 32'], inplace=True)
    
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

duplicates = data.duplicated()
print(f"Nombre de doublons : {duplicates.sum()}")


missing_values = data.isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values[missing_values > 0])

datacorr= data.corr()
columns_to_keep = datacorr.index[abs(datacorr['diagnosis']) >= 0.4]
filtered_data = data[columns_to_keep]

X = filtered_data.drop(columns=['diagnosis'])
y = filtered_data['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca2 = PCA(n_components=0.95)
X_pca2 = pca2.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca2)
pca_df['Diagnosis'] = y
pca_df['Diagnosis_FR'] = pca_df['Diagnosis']
pca_df.to_csv("X_pca.csv")



