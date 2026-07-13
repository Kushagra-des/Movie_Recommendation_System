"""
utils.py
--------
Small, reusable helper functions that are shared between preprocess.py,
recommendation.py and app.py.
"""

import os
import pickle
import re


def clean_text(text: str) -> str:
    """
    Do some very basic text cleaning:
    - Convert to lowercase
    - Remove extra spaces
    - Remove characters that are not letters, numbers or spaces

    This keeps the text vectorizer (TF-IDF) focused on meaningful words
    instead of getting confused by punctuation or inconsistent casing.

    Parameters:
        text (str): raw text (overview, keywords, etc.)

    Returns:
        str: cleaned text
    """
    if not isinstance(text, str):
        # If the value is missing (NaN) or not a string, return empty text
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()   # collapse multiple spaces
    return text


def remove_spaces(text: str) -> str:
    """
    Remove spaces inside multi-word names (e.g. cast members, director names)
    and join them into a single "token".

    Why? Because if we don't do this, TF-IDF/CountVectorizer will treat
    "Tom" and "Hanks" as two separate, generic words. This could wrongly
    match "Tom Cruise" and "Tom Hanks" as similar just because they share
    the first name "Tom". Turning "Tom Hanks" into "tomhanks" keeps each
    person/keyword as one unique unit.

    Parameters:
        text (str): a string that may contain multiple names/words

    Returns:
        str: string with spaces removed from each individual name
    """
    if not isinstance(text, str):
        return ""
    # Split on commas or extra spacing, remove inner spaces, rejoin
    parts = re.split(r"[,]", text)
    cleaned_parts = [part.strip().replace(" ", "") for part in parts if part.strip()]
    return " ".join(cleaned_parts)


def combine_features(row) -> str:
    """
    Combine the important text-based columns of a single movie row into
    one single string. This combined string is what we will later convert
    into TF-IDF vectors.

    We deliberately combine:
        genres + keywords + cast + director + overview

    Parameters:
        row (pandas.Series): one row of the movies DataFrame

    Returns:
        str: a single combined "feature soup" string for the movie
    """
    genres = remove_spaces(row.get("genres", ""))
    keywords = clean_text(row.get("keywords", ""))
    cast = remove_spaces(row.get("cast", ""))
    director = remove_spaces(row.get("director", ""))
    overview = clean_text(row.get("overview", ""))

    # Repeating genres/cast/director once more gives them slightly more
    # weight than the free-form overview text, which is a simple trick
    # often used in beginner content-based recommenders.
    combined = f"{genres} {genres} {keywords} {cast} {cast} {director} {director} {overview}"
    return clean_text(combined)


def save_pickle(obj, file_path: str) -> None:
    """
    Save any Python object to disk using Pickle.

    Parameters:
        obj: the Python object to save (DataFrame, matrix, vectorizer, etc.)
        file_path (str): where to save the .pkl file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as error:
        print(f"[ERROR] Could not save pickle file at '{file_path}': {error}")
        raise


def load_pickle(file_path: str):
    """
    Load a Python object previously saved with Pickle.

    Parameters:
        file_path (str): path to the .pkl file

    Returns:
        The unpickled Python object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Pickle file not found at '{file_path}'. "
            "Did you run 'python preprocess.py' first?"
        )
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as error:
        print(f"[ERROR] Could not load pickle file at '{file_path}': {error}")
        raise


def print_step(message: str) -> None:
    """
    Print a nicely formatted progress message to the console.
    Used while preprocessing so the user can follow along with what's
    happening (as required for a student-style project with visible logs).

    Parameters:
        message (str): the progress message to display
    """
    print(f"[INFO] {message}")
