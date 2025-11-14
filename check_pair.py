

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


OMDB_API_KEY = "ce013115"
MOVIES_CSV1 = r"d:\movie R S\movies.csv"
MOVIES_CSV2 = r"d:\movie R S\Hindi2.csv"

print("🔄 Loading and processing data...")

try:
    m1 = pd.read_csv(MOVIES_CSV1)
    m2 = pd.read_csv(MOVIES_CSV2)
except FileNotFoundError as e:
    print(f" Error: {e}")
    exit()


m2.rename(columns={'movie_name': 'title', 'genre': 'genres'}, inplace=True)
movies = pd.concat([m1, m2], ignore_index=True)
movies.drop_duplicates(subset=["title"], inplace=True, keep='first')
movies.reset_index(drop=True, inplace=True)


for col in ['genres', 'overview', 'cast', 'director']:
    if col not in movies.columns:
        movies[col] = ''
    movies[col] = movies[col].fillna('')

movies['combined'] = (
    movies['genres'].astype(str) + ' ' +
    movies['overview'].astype(str) + ' ' +
    movies['cast'].astype(str) + ' ' +
    movies['director'].astype(str)
)

tfidf = TfidfVectorizer(stop_words='english')
mat = tfidf.fit_transform(movies['combined'])
sim = linear_kernel(mat, mat)

idx_map = pd.Series(movies.index, index=movies['title']).drop_duplicates()
title_map_lower = {title.lower(): title for title in movies['title'].unique()}

print(" Data loaded and model ready.")

def get_movie_data(title):
    """Fetch poster and plot description from OMDb API."""
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}&plot=full&r=json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("Response") == "False":
            return None, f"No details found for '{title}'."
        poster = data.get("Poster") if data.get("Poster") != "N/A" else None
        desc = data.get("Plot", "No description available.")
        return poster, desc
    except requests.exceptions.RequestException as e:
        return None, f"⚠️ Connection failed: {e}"

def get_similar_movies(title, topk=10):
    """Return top similar movies based on content similarity."""
    if title not in movies['title'].values:
        return []
    idx = int(idx_map[title])
    sims = list(enumerate(sim[idx]))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    topk_list = [movies['title'].iloc[i] for i, _ in sims_sorted[1:topk+1]]
    return topk_list

if __name__ == "__main__":
    print("\n🎬 Movie Recommender (Online - OMDb API)")
    print("Type 'exit' to quit.\n")

    while True:
        movie_input = input("🎞️ Enter a movie title: ").strip()
        if movie_input.lower() == 'exit':
            break

        movie_name = title_map_lower.get(movie_input.lower())
        if not movie_name:
            print(" Movie not found in dataset. Try again.")
            continue

        recommendations = get_similar_movies(movie_name)
        poster, desc = get_movie_data(movie_name)

        print("\n" + "=" * 50)
        print(f"🎬 {movie_name}")
        print(f"Poster: {poster or 'Not Found'}")
        print(f"Description: {desc}\n")

        print("🎯 Top 10 Similar Movies:")
        for i, t in enumerate(recommendations, 1):
            print(f"{i}. {t}")
        print("=" * 50 + "\n")
