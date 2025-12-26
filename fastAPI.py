from fastapi import FastAPI, HTTPException, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import pandas as pd
import numpy as np
import requests
import re
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Request
from contextlib import asynccontextmanager

# -----------------------------
# Recommender state in memory
# -----------------------------
movies_df: pd.DataFrame | None = None
similarity: np.ndarray | None = None

def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def rebuild_from_db(db: Session):
    global movies_df, similarity

    rows = db.query(MovieDB).all()
    if not rows:
        movies_df = pd.DataFrame(columns=["movie_id", "title", "tags", "poster_path"])
        similarity = np.zeros((0, 0))
        return

    movies_df = pd.DataFrame([{
        "movie_id": r.movie_id,
        "title": r.title,
        "tags": r.tags,
        "poster_path": r.poster_path,
    } for r in rows])

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(movies_df["tags"].fillna(""))
    similarity = cosine_similarity(vectors)


def fetch_poster(movie_id: int):
    # Do NOT hardcode your TMDB key in production. Use env var.
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_TMDB_KEY&language=en-US"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    poster_path = data.get("poster_path")
    return poster_path  # store just the path; build full URL in response if you want

def recommend(movie: str):
    global movies_df, similarity
    print(f"Movies loaded: {len(movies_df) if movies_df is not None else 0}")
    if movies_df is None or similarity is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    query = (movie or "").strip().lower()
    if not query:
        raise HTTPException(status_code=400, detail="Movie name is required")

    # case-insensitive match
    titles_lower = movies_df["title"].astype(str).str.lower()
    matches = movies_df[titles_lower == query]
    if matches.empty:
        print(f"No exact match for '{query}'. Available titles: {list(movies_df['title'].head(10))}")
        raise HTTPException(status_code=404, detail="Movie not found")

    index = matches.index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recs = []
    for i in distances[1:6]:
        row = movies_df.iloc[i[0]]
        title = row["title"]
        poster_path = row.get("poster_path")

        poster_url = None
        if isinstance(poster_path, str) and poster_path.strip():
            poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path.lstrip('/')}"


        recs.append({"title": title, "poster": poster_url})

    return recs

# -----------------------------
# Lifespan for startup/shutdown
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db = SessionLocal()
    try:
        rebuild_from_db(db)
    finally:
        db.close()
    yield
    # Shutdown (if needed)

# -----------------------------
# App + CORS
# -----------------------------
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"]
,  # URL du frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Database (SQLite)
# -----------------------------
DATABASE_URL = "sqlite:///./movies.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

class MovieDB(Base):
    __tablename__ = "movies"
    movie_id = Column(Integer, primary_key=True, index=True)  # TMDB id or your id
    title = Column(String(255), unique=True, index=True, nullable=False)
    tags = Column(Text, nullable=False)  # text used by the recommender
    poster_path = Column(String(255), nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# Authentication
# -----------------------------
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
SECRET_KEY = "your-secret-key"  # use env var in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_admin(db: Session, username: str, password: str):
    admin = db.query(Admin).filter(Admin.username == username).first()
    if not admin:
        return False
    if not verify_password(password, admin.hashed_password):
        return False
    return admin

def get_current_admin(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# -----------------------------
# Pydantic Models
# -----------------------------
class AdminCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class MovieCreate(BaseModel):
    movie_id: int = Field(..., description="TMDB id or your own id")
    title: str
    tags: str = Field(..., description="Text features used by your model (overview + genres + cast + keywords...)")

class MovieUpdate(BaseModel):
    title: str
    tags: str

class LoginRequest(BaseModel):
    username: str
    password: str

# -----------------------------
# Public endpoint
# -----------------------------
@app.get("/recommend/{movie_name}")
def get_recommendations(movie_name: str):
    recs = recommend(movie_name)
    return {"recommendations": recs}

@app.get("/movies")
def get_movies(db: Session = Depends(get_db)):
    rows = db.query(MovieDB).all()
    return [{"title": r.title, "movie_id": r.movie_id} for r in rows]

# -----------------------------
# Admin endpoints
# -----------------------------
@app.post("/admin/register", response_model=Token)
def register_admin(admin: AdminCreate, db: Session = Depends(get_db)):
    db_admin = db.query(Admin).filter(Admin.username == admin.username).first()
    if db_admin:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(admin.password)
    db_admin = Admin(username=admin.username, hashed_password=hashed_password)
    db.add(db_admin)
    db.commit()
    db.refresh(db_admin)
    access_token = create_access_token(data={"sub": admin.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/admin/login", response_model=Token)
def login_admin(credentials: LoginRequest, db: Session = Depends(get_db)):
    admin = authenticate_admin(db, credentials.username, credentials.password)
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": credentials.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/admin/movies", response_model=dict)
def add_movie(payload: MovieCreate, current_admin: str = Depends(get_current_admin), db: Session = Depends(get_db)):
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    tags = clean_text(payload.tags)
    if not tags:
        raise HTTPException(status_code=400, detail="Tags are required")

    existing = db.query(MovieDB).filter(MovieDB.title == title).first()
    if existing:
        raise HTTPException(status_code=400, detail="Movie with this title already exists")

    poster_path = fetch_poster(payload.movie_id)  # optional
    movie = MovieDB(movie_id=payload.movie_id, title=title, tags=tags, poster_path=poster_path)
    db.add(movie)
    db.commit()

    # Rebuild similarity so the new movie is included immediately
    rebuild_from_db(db)

    return {"success": True, "message": "Movie added and model rebuilt"}

@app.put("/admin/movies/{movie_id}")
def update_movie(movie_id: int, payload: MovieUpdate, current_admin: str = Depends(get_current_admin), db: Session = Depends(get_db)):
    movie = db.query(MovieDB).filter(MovieDB.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    tags = clean_text(payload.tags)
    if not tags:
        raise HTTPException(status_code=400, detail="Tags are required")

    movie.title = title
    movie.tags = tags
    db.commit()

    # Rebuild similarity
    rebuild_from_db(db)

    return {"success": True, "message": "Movie updated and model rebuilt"}

@app.delete("/admin/movies/{movie_id}")
def delete_movie(movie_id: int, current_admin: str = Depends(get_current_admin), db: Session = Depends(get_db)):
    movie = db.query(MovieDB).filter(MovieDB.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    db.delete(movie)
    db.commit()

    # Rebuild similarity
    rebuild_from_db(db)

    return {"success": True, "message": "Movie deleted and model rebuilt"}

@app.get("/admin/movies")
def list_movies(current_admin: str = Depends(get_current_admin), db: Session = Depends(get_db)):
    movies = db.query(MovieDB).all()
    return [{"movie_id": m.movie_id, "title": m.title, "tags": m.tags[:100] + "..." if len(m.tags) > 100 else m.tags} for m in movies]
