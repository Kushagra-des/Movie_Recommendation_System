"""
add_movie.py
------------
A small helper script for adding new movies to dataset/movies.csv.

Why this file exists:
    Adding a row directly to a CSV by hand is easy to get wrong (missing
    commas, wrong column order, forgetting to re-run preprocessing).
    This script does it safely: it validates the input, appends a new
    row with the correct columns, avoids duplicate titles, and then
    automatically re-runs the preprocessing pipeline so the new movie
    is immediately available to the Streamlit app.

Two ways to use it:

1) INTERACTIVE (recommended for beginners) - just run it and answer
   the prompts:
       python add_movie.py

2) PROGRAMMATIC - import add_movie() and call it from your own code:
       from add_movie import add_movie
       add_movie(
           title="Inception",
           genres="Action Science Fiction Thriller",
           keywords="dream heist subconscious",
           overview="A thief who steals corporate secrets through dream-sharing technology...",
           cast="Leonardo DiCaprio Joseph Gordon-Levitt Ellen Page",
           director="Christopher Nolan",
           popularity=80.0,
           vote_average=8.3,
           release_date="16-07-2010",
       )
"""

import os

import pandas as pd

from preprocess import run_preprocessing_pipeline
from utils import print_step

DATASET_PATH = "dataset/movies.csv"

# The exact column order used in dataset/movies.csv.
# Keeping this consistent avoids silently misaligned columns.
COLUMNS = [
    "id", "title", "genres", "keywords", "overview",
    "cast", "director", "popularity", "vote_average", "release_date",
]


def _generate_new_id(df: pd.DataFrame) -> int:
    """Generate a new unique movie id (one higher than the current max)."""
    if df.empty:
        return 1
    return int(df["id"].max()) + 1


def add_movie(
    title: str,
    genres: str = "",
    keywords: str = "",
    overview: str = "",
    cast: str = "",
    director: str = "",
    popularity: float = 0.0,
    vote_average: float = 0.0,
    release_date: str = "",
    auto_preprocess: bool = True,
) -> bool:
    """
    Add a single new movie to dataset/movies.csv.

    Parameters:
        title (str): Movie title (required)
        genres (str): Space-separated genres, e.g. "Action Comedy"
        keywords (str): Space-separated plot keywords
        overview (str): A short plot summary
        cast (str): Space-separated lead actor names, e.g. "Tom Hanks Meg Ryan"
        director (str): Director's name
        popularity (float): Popularity score (any positive number is fine)
        vote_average (float): Rating out of 10
        release_date (str): Format DD-MM-YYYY, to match the existing dataset
        auto_preprocess (bool): If True, automatically re-runs preprocess.py
                                 after adding the movie so it's instantly
                                 usable in the app.

    Returns:
        bool: True if the movie was added successfully, False otherwise.
    """
    if not title or not title.strip():
        print("[ERROR] A movie title is required.")
        return False

    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Could not find dataset at '{DATASET_PATH}'.")
        return False

    # Avoid adding a movie that already exists (case-insensitive check)
    if title.strip().lower() in df["title"].str.lower().values:
        print(f"[WARNING] '{title}' already exists in the dataset. Skipping.")
        return False

    new_row = {
        "id": _generate_new_id(df),
        "title": title.strip(),
        "genres": genres.strip(),
        "keywords": keywords.strip(),
        "overview": overview.strip(),
        "cast": cast.strip(),
        "director": director.strip(),
        "popularity": popularity,
        "vote_average": vote_average,
        "release_date": release_date.strip(),
    }

    df = pd.concat([df, pd.DataFrame([new_row])[COLUMNS]], ignore_index=True)

    try:
        df.to_csv(DATASET_PATH, index=False)
        print_step(f"Added '{title}' to {DATASET_PATH}.")
    except Exception as error:
        print(f"[ERROR] Could not save the updated dataset: {error}")
        return False

    if auto_preprocess:
        print_step("Re-running preprocessing so the new movie is usable immediately...")
        run_preprocessing_pipeline()

    return True


def _run_interactive_mode():
    """Ask the user questions in the terminal and add the movie they describe."""
    print("=== Add a New Movie ===")
    print("(Press Enter to leave a field blank where optional)\n")

    title = input("Title (required): ").strip()
    genres = input("Genres (space-separated, e.g. 'Action Comedy'): ").strip()
    keywords = input("Keywords (space-separated): ").strip()
    overview = input("Overview / plot summary: ").strip()
    cast = input("Cast (space-separated, e.g. 'Tom Hanks Meg Ryan'): ").strip()
    director = input("Director: ").strip()

    popularity_input = input("Popularity score (number, optional): ").strip()
    popularity = float(popularity_input) if popularity_input else 0.0

    vote_input = input("Vote average out of 10 (optional): ").strip()
    vote_average = float(vote_input) if vote_input else 0.0

    release_date = input("Release date (DD-MM-YYYY, optional): ").strip()

    success = add_movie(
        title=title,
        genres=genres,
        keywords=keywords,
        overview=overview,
        cast=cast,
        director=director,
        popularity=popularity,
        vote_average=vote_average,
        release_date=release_date,
        auto_preprocess=True,
    )

    if success:
        print(f"\n✅ '{title}' was added successfully! Restart the Streamlit app to see it.")
    else:
        print(f"\n❌ Could not add '{title}'. See the messages above for details.")


if __name__ == "__main__":
    _run_interactive_mode()
