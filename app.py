import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

OMDB_API_KEY = "ce013115"

@st.cache_data
def load_movies():
    try:
        m1 = pd.read_csv("movies.csv")
        m2 = pd.read_csv("Hindi2.csv")
    except FileNotFoundError as e:
        st.error(f" {e}")
        st.stop()

    m2.rename(columns={"movie_name": "title", "genre": "genres", "lead_actor": "cast"}, inplace=True)
    movies = pd.concat([m1, m2], ignore_index=True).drop_duplicates(subset=["title"]).reset_index(drop=True)
    
    for c in ["genres", "overview", "cast", "director"]:
        if c not in movies.columns:
            movies[c] = ""
        movies[c] = movies[c].fillna("")
    
    movies["combined"] = movies["genres"] + " " + movies["overview"] + " " + movies["cast"] + " " + movies["director"]
    return movies


movies = load_movies()
tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(movies["combined"])
cosine_sim = linear_kernel(matrix, matrix)
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def get_movie_data(title):
    """Fetch movie poster and description from OMDb API."""
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}&plot=full&r=json"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("Response") == "False":
            return None, f"No details found for '{title}'."
        poster = data.get("Poster") if data.get("Poster") != "N/A" else None
        plot = data.get("Plot", "No description available.")
        return poster, plot
    except requests.exceptions.RequestException:
        return None, "⚠️ Connection failed."

def recommend(title, num=6):
    """Recommend top similar movies using TF-IDF + Cosine Similarity."""
    if title not in indices:
        return []
    idx = indices[title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num + 1]
    return [(movies["title"].iloc[i], s) for i, s in scores]

st.set_page_config(page_title="🎬 AI Movie Recommendation System", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #002244;
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }
    .movie-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    .movie-title {
        font-size: 18px;
        font-weight: bold;
        margin: 10px 0 5px 0;
        color: #FFD700;
    }
    .movie-desc {
        font-size: 14px;
        color: #E0E0E0;
        text-align: justify;
    }
    div.stButton > button:first-child {
        background-color: black;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: 2px solid red;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: red;
        color: white;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🎞️ Navigation")
mode = st.sidebar.radio("Select Mode:", ["🎬 Recommend Movies", "⚖️ Compare Two Movies"])

if mode == "🎬 Recommend Movies":
    st.title("🍿 Movie Recommendation System (OMDb API)")
    st.write("Get similar movies with **posters and descriptions**, powered by content-based filtering.")

    movie_choice = st.selectbox("🎥 Choose a movie:", movies["title"].values)

    if st.button("🔍 Recommend"):
        recs = recommend(movie_choice)
        if not recs:
            st.warning("No recommendations found.")
        else:
            cols = st.columns(3)
            for i, (t, s) in enumerate(recs):
                with cols[i % 3]:
                    poster, desc = get_movie_data(t)
                    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                    st.image(poster or "https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
                    st.markdown(f"<div class='movie-title'>{t}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='movie-desc'>{desc}<br><br><b>Similarity:</b> {s:.2%}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "⚖️ Compare Two Movies":
    st.title("⚖️ Compare Two Movies")
    st.write("Compare the similarity between two movies and view their descriptions side by side.")

    movie_a = st.selectbox("🎬 Select the first movie:", movies["title"].values)
    movie_b = st.selectbox("🎬 Select the second movie:", movies["title"].values, index=1)

    if st.button("Compare"):
        if movie_a not in indices or movie_b not in indices:
            st.error(" One or both movies not found.")
        else:
            idx_a, idx_b = indices[movie_a], indices[movie_b]
            score = cosine_sim[idx_a, idx_b]

            st.subheader(f"🔍 Comparison: '{movie_a}' vs '{movie_b}'")
            st.metric("Similarity Score", f"{score:.2%}")

            poster_a, desc_a = get_movie_data(movie_a)
            poster_b, desc_b = get_movie_data(movie_b)

            col1, col2 = st.columns(2)
            with col1:
                st.image(poster_a or "https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
                st.write(f"**{movie_a}**: {desc_a}")
            with col2:
                st.image(poster_b or "https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
                st.write(f"**{movie_b}**: {desc_b}")

            st.info(f"🎯 '{movie_b}' has a similarity of **{score:.2%}** with '{movie_a}'.")
