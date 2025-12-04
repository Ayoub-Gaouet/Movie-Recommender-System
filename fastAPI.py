from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import requests
import pandas as pd

app = FastAPI()

# Allow CORS for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" by your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model and data
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def fetch_poster(movie_id: int):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return f"https://image.tmdb.org/t/p/w500/{data.get('poster_path', '')}"

def recommend(movie: str):
    if movie not in movies['title'].values:
        raise HTTPException(status_code=404, detail="Movie not found")

    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended = []
    for i in distances[1:6]:
        movie_id = int(movies.iloc[i[0]].movie_id)
        title = movies.iloc[i[0]].title
        poster = fetch_poster(movie_id)
        recommended.append({"title": title, "poster": poster})

    return recommended

@app.get("/recommend/{movie_name}")
def get_recommendations(movie_name: str):
    recs = recommend(movie_name)
    return {"recommendations": recs}
