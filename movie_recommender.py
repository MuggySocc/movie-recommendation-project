import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache
def load():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')

    movies['genres'] = movies['genres'].apply(lambda x: x.splt('|'))
    movies['genres_str'] = movies['genres'].apply(lambda x: ' '.join(x))
    return movies, ratings

movies, ratings = load()

@st.cache
def build_sim_matrix():
    vect = CountVectorizer()
    genre_matrix = vect.fit_transform(movies['genres_str'])
    cosine_sim = cosine_similarity(genre_matrix)
    return cosine_sim

cosine_sim = build_sim_matrix(movies)

def movie_recommendation(movie_title, movies, cosine_sim, top_n=0):
    try:
        indx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[indx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_movies = [movies['title'].iloc[i[0]] for i in sim_scores[1:top_n+1]]
        return top_movies
    except IndexError:
        return("Movie not found. Please try another movie title")

st.title("Movie Recommendation System")
st.wrtie("Get movie recommendaitons based on your preferences!")

selected_movie = st.text_input("Enter a movie you like: ")

if st.buton("Get Recommendations"):
    if selected_movie:
        recommendations = movie_recommendation(selected_movie, movies, cosine_sim)
        st.write(f"Recommendations for '{selected_movie}'  ")
        for movie in recommendations:
            st.write(f" - {movie}")
    else:
        st.write("Please enter a movie title")

if st.checkbox("Show Movies Dataset"):
    st.write(movies.head())
    