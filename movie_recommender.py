import tkinter as tk
from tkinter import messagebox, Listbox
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')

    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
    movies['genres_str'] = movies['genres'].apply(lambda x: ' '.join(x))
    return movies, ratings

movies, ratings = load()


def build_sim_matrix(movies):
    vect = CountVectorizer()
    genre_matrix = vect.fit_transform(movies['genres_str'])
    cosine_sim = cosine_similarity(genre_matrix)
    return cosine_sim

cosine_sim = build_sim_matrix(movies)

def movie_recommendation(movie_title, movies, cosine_sim, top_n=10):
    matches = movies[movies['title'].str.contains(movie_title, case=False, na=False)]
    if matches.empty:
        return f"No matches found for '{movie_title}'"
    
    recommendation_list = []
    for indx in matches.index:
        sim_scores = list(enumerate(cosine_sim[indx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        recommendation_list.extend(movies['title'].iloc[i[0]] for i in sim_scores[1:top_n+1])

    recommendations_list = list(dict.fromkeys(recommendation_list))

    return recommendations_list

def get_recommendations():
    movie_title = entry.get()
    if not movie_title:
        messagebox.showwarning("Input Required", "Please enter a movie title!")
        return
    
    results = movie_recommendation(movie_title, movies, cosine_sim)
    if isinstance(results, str):
        messagebox.showinfo("Results", results)
    else:
        listbox.delete(0, tk.END)
        for movie in results:
            listbox.insert(tk.END, movie)

root = tk.Tk()
root.title("Movie Recommender")

frame = tk.Frame(root)
frame.pack(pady=10)
label = tk.Label(frame, text="Enter movie title:")
label.pack(side=tk.LEFT, padx=5)
entry = tk.Entry(frame, width=50)
entry.pack(side=tk.LEFT, padx=5)
btn = tk.Button(frame, text="Recommend", command=get_recommendations)
btn.pack(side=tk.LEFT, padx=5)

listbox = Listbox(root, width=80, height=20)
listbox.pack(pady=10)

root.mainloop()
