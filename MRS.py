

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans


print("🔄 Loading and processing data...")

MOVIES_CSV1 = r"d:\movie R S\movies.csv"
MOVIES_CSV2 = r"d:\movie R S\Hindi2.csv"

movies1 = pd.read_csv(MOVIES_CSV1)
movies2 = pd.read_csv(MOVIES_CSV2)

movies2.rename(columns={'movie_name': 'title', 'genre': 'genres'}, inplace=True)

movies = pd.concat([movies1, movies2], ignore_index=True)
movies.drop_duplicates(subset=["title"], inplace=True)
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

print("Data loaded. TF-IDF + Cosine Similarity model ready.")


print("🔍 Performing K-Means clustering...")

num_clusters = 6  # you can change (e.g., 5 or 8)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
movies['cluster'] = kmeans.fit_predict(mat)

print(f" Clustering complete. Movies grouped into {num_clusters} clusters.")


def get_similar_movies(title, topk=10):
    if title not in movies['title'].values:
        return None, []
    idx = int(idx_map[title])
    sims = list(enumerate(sim[idx]))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    # Skip itself and return top similar movies
    topk_list = [movies['title'].iloc[i] for i, s in sims_sorted[1:topk+1]]
    return topk_list


def show_cluster_movies(title):
    if title not in movies['title'].values:
        return []
    cluster_id = movies.loc[movies['title'] == title, 'cluster'].values[0]
    cluster_movies = movies[movies['cluster'] == cluster_id]['title'].tolist()
    cluster_movies.remove(title)
    return cluster_id, cluster_movies


if __name__ == "__main__":
    print("\n🎬 Movie Recommender + Clustering CLI")
    print("Type 'exit' to quit.\n")

    while True:
        movie_input = input("Enter a movie title: ").strip()
        if movie_input.lower() == 'exit':
            break

        movie_name = title_map_lower.get(movie_input.lower())

        if not movie_name:
            print("❌ Movie not found. Try again.\n")
            continue

       
        cluster_id, cluster_movies = show_cluster_movies(movie_name)
        print(f"\n🧠 '{movie_name}' belongs to Cluster {cluster_id}")
        print(f"🎞 Other movies in the same cluster:")
        for m in cluster_movies[:10]:
            print(f"  - {m}")

        
        recommendations = get_similar_movies(movie_name)
        print("\n" + "=" * 50)
        print(f"🎬 Top 10 similar movies to '{movie_name}':")
        for i, title in enumerate(recommendations, 1):
            print(f"{i}. {title}")
        print("=" * 50 + "\n")
