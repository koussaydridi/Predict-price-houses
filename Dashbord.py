import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

st.title("Analyse et Prédiction des Prix Immobiliers")


st.write("### 10 ligne de notre Base :")
data=pd.read_csv('train.csv')
st.write(data.head(10))




# Calculer les informations sur les colonnes
st.write("### Aperçu des colonnes et des types de données :")
info = pd.DataFrame({
    "Non-Null Count": data.notnull().sum(),
    "Dtype": data.dtypes,
})

info["Highlight"] = info["Non-Null Count"] < len(data)


# Afficher le tableau avec mise en évidence
st.dataframe(info.style.applymap(
    lambda x: "background-color: red" if x and isinstance(x, bool) else "",
    subset=["Highlight"]
))

numeric_cols = data.select_dtypes(include=["float", "int"]).columns
means = data[numeric_cols].mean()
data[numeric_cols] = data[numeric_cols].fillna(means)

for column in data.columns:
    if data[column].isnull().any():
        mode_value = data[column].mode().iloc[0]
        data[column].fillna(mode_value, inplace=True)

# Select categorical columns
categorical_cols = data.select_dtypes(include=["object"]).columns
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Apply label encoding to categorical columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])


st.write("### Description sur DB :")
st.write(data.describe())

st.write("### la corrélation avec la cible 'SalePrice'")
count_columns = len(data.columns)
st.write("*", count_columns, "*")
correlation = data.corr()["SalePrice"].sort_values(ascending=False)
st.write(correlation)



# Calculer la corrélation avec la colonne 'SalePrice'
correlation = data.corr()["SalePrice"].sort_values(ascending=False)
correlation_df = pd.DataFrame(correlation).reset_index()
correlation_df.columns = ['Feature', 'Correlation']


fig = px.bar(
    correlation_df,
    x='Feature',
    y='Correlation',
    color='Correlation',
    color_continuous_scale='RdBu',
    labels={'Correlation': 'Corrélation', 'Feature': 'Caractéristiques'},
    title="Corrélation avec SalePrice"
)
st.plotly_chart(fig)

st.write("### Nouvelle données")
data = data[[col for col in data.columns if col != 'SalePrice'] + ['SalePrice']]

st.write(data.head(10))



fig = px.histogram(
    data_frame=data, 
    x="SalePrice", 
    nbins=50,  # Nombre de bacs (bins) dans l'histogramme
    title="Distribution des prix de vente",
    labels={"SalePrice": "Prix de vente"},  # Label pour l'axe X
    opacity=0.75  # Opacité des barres
)

st.plotly_chart(fig)

correlation=data.corr()
plt.figure(figsize=(50,50))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
st.pyplot(plt)





OverallQual_price = data.groupby('OverallQual')['SalePrice'].mean()
st.write(OverallQual_price)


# Créer un nuage de points pour afficher la relation entre la rue et le prix de vente
plt.figure(figsize=(12, 6))
sns.scatterplot(x='OverallQual', y='SalePrice', data=data, hue='OverallQual', palette='Set2', s=100, alpha=0.7)

# Ajouter des titres et labels
plt.title('Relation entre la Qualiter de maison et le Prix de Vente', fontsize=16)
plt.xlabel('Qauliter de maison ', fontsize=12)
plt.ylabel('Prix de Vente', fontsize=12)

# Rotation des étiquettes de l'axe x pour éviter qu'elles se chevauchent
plt.xticks(rotation=45)
st.pyplot(plt)


X=data.drop('SalePrice',axis=1)
y= data['SalePrice'].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compilation du modèle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entraînement
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32)

# Évaluation
test_loss, test_mae = model.evaluate(X_test, y_test)
st.write(f"Erreur absolue moyenne sur l'ensemble de test : {test_mae}")


st.write("### Prédictions")
y_pred = model.predict(X_test)

# Évaluation des performances du meilleur modèle
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred))
mae_best = mean_absolute_error(y_test, y_pred)
r2_best = r2_score(y_test, y_pred)
# Affichage des résultats

st.write("\nPerformance du modèle optimisé :")
st.write(f"RMSE sur les données de test : {rmse_best:.2f}")
st.write(f"MAE sur les données de test : {mae_best:.2f}")
st.write(f"Score R² sur les données de test : {r2_best:.2f}")

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Ligne idéale (valeurs réelles = prédictions)')
plt.title('Valeurs réelles vs Prédictions')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.grid(True)
plt.legend()
st.pyplot(plt)

st.write("### Nouvelle données")


data_test=pd.read_csv('test.csv')
st.write(data_test.head(10))

st.write("### Aperçu des colonnes et des types de données test:")
info_test = pd.DataFrame({
    "Non-Null Count": data_test.notnull().sum(),
    "Dtype": data_test.dtypes,
})

info_test["Highlight"] = info_test["Non-Null Count"] < len(data_test)


# Afficher le tableau avec mise en évidence
st.dataframe(info.style.applymap(
    lambda x: "background-color: red" if x and isinstance(x, bool) else "",
    subset=["Highlight"]
))

st.write("### Crée un Pipeline pour faire traitement au base de test")

# Séparer les colonnes numériques et catégorielles
categ_columns = data_test.select_dtypes(include=['object']).columns
numeric_columns = data_test.select_dtypes(include=['int64', 'float64']).columns

# Pipeline pour les colonnes numériques
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline pour les colonnes catégorielles avec OneHotEncoder
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# ColumnTransformer pour appliquer les transformations
processing = ColumnTransformer([
    ('cat', cat_pipeline, categ_columns),
    ('num', num_pipeline, numeric_columns)
])

# Appliquer le traitement
data_processed = processing.fit_transform(data_test)

# Convertir en DataFrame pour un affichage plus facile
all_columns = list(categ_columns) + list(numeric_columns)
data_processed_df = pd.DataFrame(data_processed, columns=all_columns)

lp=LabelEncoder()
data_new=data_processed_df.apply(lp.fit_transform)
st.write("### Nouvelle donnée de test")
st.write(data_new.head(5))

st.write("### Les Prédictions")

y_pred = model.predict(data_new)

predictions_df = pd.DataFrame(y_pred, columns=['Prédictions'])# Afficher le tableau interactif dans Streamlit
st.dataframe(predictions_df)
