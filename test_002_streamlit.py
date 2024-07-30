import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from fuzzywuzzy import process

# Charger le DataFrame
df = pd.read_csv('df_streamlit.csv')

# Vérifier les types des colonnes
print(df.dtypes)

# Sélectionner uniquement les colonnes numériques pour la normalisation
numeric_cols = df.select_dtypes(include='number').columns
df_numeric = df[numeric_cols]

# Normaliser les données numériques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Créer la liste de films
list_movies = list(df['primaryTitle'])

# Charger le modèle KNN
knn_model = joblib.load('knn_model_streamlit.joblib')

# Fonction pour trouver le film avec une correction d'erreurs
def trouver_titre_film(entree_utilisateur, liste_film):
    meilleur_match = process.extractOne(entree_utilisateur, liste_film)
    return meilleur_match[0]

# Fonction de recommandation
def recommend_movies(movie_title):
    movie_index = df[df['primaryTitle'] == movie_title].index[0]
    _, indices = knn_model.kneighbors([X_scaled[movie_index]])

    # Récupération des indices des films recommandés (en excluant le film d'origine)
    recommended_movie_indices = indices[0][1:]

    # Récupération des titres des films recommandés
    recommended_movies = [df['primaryTitle'][index] for index in recommended_movie_indices]
    
    return recommended_movies

# Configuration de l'application Streamlit
st.title("Système de recommandation de films")
st.write("Entrez le titre d'un film pour obtenir des recommandations de films similaires.")

# Entrée utilisateur
user_input = st.text_input("Titre du film")

if user_input:
    # Trouver le film avec la correction d'erreurs
    corrected_title = trouver_titre_film(user_input, list_movies)
    st.write(f"Vous avez voulu dire : {corrected_title}")

    # Afficher les recommandations
    recommendations = recommend_movies(corrected_title)
    st.write("Recommandations de films :")
    for movie in recommendations:
        st.write(movie)
