import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


vect = CountVectorizer()
scaler = MinMaxScaler()
mlb = MultiLabelBinarizer()

links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

#Preprocessing Data

#print(movies.head())
#print(ratings.head())

#print(movies.info())
#print(ratings.info())

#print(movies.isnull().sum())
#print(ratings.isnull().sum())

##Ratings Data Distribution
#print(ratings['rating'].describe())
#print(ratings['userId'].nunique())
#print(ratings['movieId'].nunique())

##Movies Data Distribution
#print(movies['genres'].value_counts())

#Dropping Missing Values
#movies.dropna(inplace=True)
#ratings.dropna(inplace=True)

#Changing The Genres in movies['genres'] into lists dividing at the '|' (This will change depending on data sheet)
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

merged_data = pd.merge(ratings, movies, on='movieId')
#print(merged_data.head())

ratings_sum = ratings.groupby('movieId').agg({
    'rating': ['mean','count']
}).reset_index()
# This sets the movieId to show as column when displaying
ratings_sum.columns = ['moviesId', 'avg_rating', 'num_ratings']
#print(ratings_sum.head())

genres_encoded = pd.DataFrame(mlb.fit_transform(movies['genres']),
                              columns=mlb.classes_,
                              index=movies.index)
movies = pd.concat([movies, genres_encoded], axis = 1)
#print(movies.head())

#Normalizing the ratings to a standard scale
ratings['rating'] = scaler.fit_transform(ratings[['rating']])

#Splitting data from training and testing
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

#Combines movie genres into a string
movies['genres_str'] = movies['genres'].apply(lambda x: ' '.join(x))
#print(movies[['title', 'genres_str']].head())

#Using count vectorizer to convert text into a 'bog of word' representation
genre_matrix = vect.fit_transform(movies['genres_str'])
#print(genre_matrix.shape)

#This compares cosine similarity to measure similarity between to vectors
cosine_sim = cosine_similarity(genre_matrix)
#print(cosine_sim)



#Recommendation Functionaility 
def recommend(movie_title, movies, cosine_sim):
    #Finds the index of the movie in the datasheet
    idx = movies[movies['title'] == movie_title].index[0]
    #Pulls similarity in scores for the movies using index made before
    sim_scores = list(enumerate(cosine_sim[idx]))
    #This sorts the similarity in decending orders (Gonna be honest idk what this does fully. Look up more info)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse = True)
    #This creates top 10 similar movies minus the first option becasue that is the selected movie
    top_movies = [movies['title'].iloc[i[0]] for i in sim_scores[1:11]]
    return top_movies

movies =  pd.merge(movies, ratings_sum, on='movieId', how='left')
movies['avg_rating'].fillna(0, inplace = True) # Fills any missing info defined by 'avg_rating' with 0 (if any data is missing)