import pickle
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# Database setup (same as fastAPI.py)
DATABASE_URL = "sqlite:///./movies.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class MovieDB(Base):
    __tablename__ = "movies"
    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), unique=True, index=True, nullable=False)
    tags = Column(Text, nullable=False)
    poster_path = Column(String(255), nullable=True)

Base.metadata.create_all(bind=engine)

def populate_db():
    # Load the pickled dataframe
    movies_df = pickle.load(open('movie_list.pkl', 'rb'))

    # Remove duplicates based on title
    movies_df = movies_df.drop_duplicates(subset=['title'])

    db = SessionLocal()
    try:
        # Drop and recreate table
        MovieDB.__table__.drop(engine, checkfirst=True)
        MovieDB.__table__.create(engine)
        print("Recreated movies table.")

        for _, row in movies_df.iterrows():
            movie = MovieDB(
                movie_id=row['movie_id'],
                title=row['title'],
                tags=row['tags'],
                poster_path=None  # Will be fetched later if needed
            )
            db.add(movie)
        db.commit()
        print("Database populated successfully.")
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    populate_db()
