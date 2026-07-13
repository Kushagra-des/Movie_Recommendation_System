"""
recommendation.py
------------------
This file contains the core recommendation logic:

- Loading the pre-processed movie data and similarity matrix (created by
  preprocess.py)
- Finding the closest matching movie title, even if the user made a
  small spelling mistake (using difflib's fuzzy matching)
- Returning the Top-N most similar movies for a given movie title

This module is used by app.py (the Streamlit UI), but it can also be
tested/run independently from the command line for quick debugging.
"""

import difflib
import pandas as pd
from utils import load_pickle, print_step

MOVIES_PKL_PATH = "models/movies_data.pkl"
SIMILARITY_PKL_PATH = "models/similarity.pkl"


def load_model_data():
    """
    Load the processed movies DataFrame and the cosine similarity matrix
    from disk (created earlier by preprocess.py).

    Returns:
        tuple: (movies_df, similarity_matrix)
    """
    movies_df = load_pickle(MOVIES_PKL_PATH)
    similarity_matrix = load_pickle(SIMILARITY_PKL_PATH)
    return movies_df, similarity_matrix


def get_all_titles(movies_df: pd.DataFrame) -> list:
    """
    Return a plain Python list of all movie titles in the dataset.
    Useful for populating dropdowns/search boxes in the Streamlit app.

    Parameters:
        movies_df (pd.DataFrame): the processed movies dataset

    Returns:
        list[str]: all movie titles, sorted alphabetically
    """
    return sorted(movies_df["title"].dropna().unique().tolist())


def find_closest_match(movie_name: str, all_titles: list) -> str | None:
    """
    Find the closest matching movie title using difflib's fuzzy matching.
    This allows the app to handle small spelling mistakes gracefully,
    e.g. typing "Avngers" should still find "The Avengers".

    Parameters:
        movie_name (str): the (possibly misspelled) movie name typed by the user
        all_titles (list[str]): list of all valid movie titles in the dataset

    Returns:
        str | None: the closest matching title, or None if nothing close enough was found
    """
    if not movie_name or not movie_name.strip():
        return None

    close_matches = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.5)

    if close_matches:
        return close_matches[0]
    return None


def get_recommendations(movie_name: str, movies_df: pd.DataFrame, similarity_matrix, top_n: int = 10) -> dict:
    """
    Generate the Top-N movie recommendations for a given movie name.

    Steps:
        1. Find the closest matching movie title (handles typos).
        2. Look up that movie's index in the DataFrame.
        3. Get its similarity scores against every other movie.
        4. Sort by similarity score (highest first) and take the top N,
           excluding the movie itself.
        5. Return a friendly dictionary with the results (or an error message).

    Parameters:
        movie_name (str): movie title typed by the user
        movies_df (pd.DataFrame): processed movies dataset
        similarity_matrix: cosine similarity matrix from preprocess.py
        top_n (int): how many recommendations to return (default 10)

    Returns:
        dict: {
            "success": bool,
            "matched_title": str or None,
            "message": str,
            "recommendations": list[dict]  # empty if not successful
        }
    """
    try:
        all_titles = get_all_titles(movies_df)
        matched_title = find_closest_match(movie_name, all_titles)

        # Friendly message if we truly can't find anything close
        if matched_title is None:
            return {
                "success": False,
                "matched_title": None,
                "message": (
                    f"Sorry, we couldn't find any movie similar to '{movie_name}' "
                    "in our dataset. Please check the spelling and try again."
                ),
                "recommendations": [],
            }

        # Find the row index of the matched movie
        movie_index = movies_df[movies_df["title"] == matched_title].index[0]

        # Get similarity scores of this movie with every other movie
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))

        # Sort movies based on similarity score, in descending order
        sorted_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)

        # Skip the first result because it will always be the movie itself
        top_matches = sorted_scores[1: top_n + 1]

        recommendations = []
        for rank, (index, score) in enumerate(top_matches, start=1):
            movie_row = movies_df.iloc[index]
            recommendations.append(
                {
                    "rank": rank,
                    "title": movie_row["title"],
                    "genres": movie_row.get("genres", ""),
                    "overview": movie_row.get("overview", ""),
                    "director": movie_row.get("director", ""),
                    "cast": movie_row.get("cast", ""),
                    "release_date": movie_row.get("release_date", ""),
                    "vote_average": movie_row.get("vote_average", ""),
                    "similarity_score": round(float(score) * 100, 2),  # as a percentage
                }
            )

        return {
            "success": True,
            "matched_title": matched_title,
            "message": f"Showing recommendations based on '{matched_title}'.",
            "recommendations": recommendations,
        }

    except Exception as error:
        # Catch-all safety net so the Streamlit app never crashes ungracefully
        print(f"[ERROR] Something went wrong while generating recommendations: {error}")
        return {
            "success": False,
            "matched_title": None,
            "message": f"An unexpected error occurred: {error}",
            "recommendations": [],
        }


if __name__ == "__main__":
    # Small command-line test so this file can be run/debugged on its own:
    #     python recommendation.py
    print_step("Loading model data for a quick test run...")
    movies_data, similarity = load_model_data()

    test_movie = "Avatar"
    print_step(f"Testing recommendations for: '{test_movie}'")
    result = get_recommendations(test_movie, movies_data, similarity, top_n=5)

    print(result["message"])
    for rec in result["recommendations"]:
        print(f"{rec['rank']}. {rec['title']}  (similarity: {rec['similarity_score']}%)")
