"""
preprocess.py
-------------
This script is responsible for ALL the data preparation work:

1. Load the raw movies dataset (CSV file)
2. Handle missing values
3. Select only the useful columns for content-based filtering
4. Combine important text features (genres, keywords, cast, director, overview)
5. Convert the combined text into TF-IDF vectors
6. Compute a Cosine Similarity matrix between every pair of movies
7. Save the processed DataFrame and similarity matrix as .pkl files
   so that app.py / recommendation.py can load them instantly, without
   repeating this (slightly slow) work every time the app starts.

Run this file once, from the project's root folder, using:
    python preprocess.py

It will create:
    models/movies_data.pkl      -> cleaned DataFrame
    models/similarity.pkl       -> cosine similarity matrix (numpy array)
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import combine_features, save_pickle, print_step

# ----------------------------- CONFIG --------------------------------- #
DATASET_PATH = "dataset/movies.csv"
MOVIES_PKL_PATH = "models/movies_data.pkl"
SIMILARITY_PKL_PATH = "models/similarity.pkl"

# Columns we actually need for a content-based recommender.
# (The raw CSV may contain extra columns we don't care about.)
USEFUL_COLUMNS = [
    "id",
    "title",
    "genres",
    "keywords",
    "overview",
    "cast",
    "director",
    "popularity",
    "vote_average",
    "release_date",
]

# Text columns that must never contain NaN, since they get combined into
# one big string later. We fill missing values in these with "".
TEXT_COLUMNS_TO_FILL = ["genres", "keywords", "overview", "cast", "director"]


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the movies dataset from a CSV file.

    Parameters:
        path (str): path to the CSV dataset

    Returns:
        pd.DataFrame: the raw dataset
    """
    try:
        df = pd.read_csv(path)
        print_step(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at '{path}'. Please check the file path.")
        raise
    except Exception as error:
        print(f"[ERROR] Something went wrong while loading the dataset: {error}")
        raise


def select_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns that are actually needed for the recommender.
    Dropping unused columns keeps the DataFrame (and the pickle file) small.

    Parameters:
        df (pd.DataFrame): the raw dataset

    Returns:
        pd.DataFrame: dataset with only the useful columns
    """
    available_columns = [col for col in USEFUL_COLUMNS if col in df.columns]
    missing_columns = set(USEFUL_COLUMNS) - set(available_columns)

    if missing_columns:
        print(f"[WARNING] These expected columns were not found and will be skipped: {missing_columns}")

    df = df[available_columns].copy()
    print_step(f"Selected useful columns: {available_columns}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing (NaN) values in the dataset.

    - Text columns (genres, keywords, overview, cast, director) are
      filled with an empty string, since we will be combining them into
      one text blob anyway.
    - Rows with a missing title are dropped, since a movie without a
      title is useless for recommendations.

    Parameters:
        df (pd.DataFrame): dataset that may contain missing values

    Returns:
        pd.DataFrame: dataset with missing values handled
    """
    before_rows = df.shape[0]

    # Drop rows where the title itself is missing
    df = df.dropna(subset=["title"])

    # Fill missing text fields with empty strings
    for column in TEXT_COLUMNS_TO_FILL:
        if column in df.columns:
            df[column] = df[column].fillna("")

    # Fill any remaining missing numeric fields with 0
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    after_rows = df.shape[0]
    print_step(f"Handled missing values. Rows before: {before_rows}, after: {after_rows}.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate movie titles, keeping only the first occurrence.
    Duplicate titles can confuse the recommendation function later
    (e.g. two "King Kong" entries), so we keep the dataset clean.

    Parameters:
        df (pd.DataFrame): dataset possibly containing duplicate titles

    Returns:
        pd.DataFrame: dataset without duplicate titles
    """
    before_rows = df.shape[0]
    df = df.drop_duplicates(subset="title", keep="first")
    df = df.reset_index(drop=True)
    after_rows = df.shape[0]
    print_step(f"Removed duplicate titles. Rows before: {before_rows}, after: {after_rows}.")
    return df


def create_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a new column called 'combined_features' that merges genres,
    keywords, cast, director and overview into one text string per movie.
    This combined text is what TF-IDF will actually be trained on.

    Parameters:
        df (pd.DataFrame): cleaned dataset

    Returns:
        pd.DataFrame: dataset with an added 'combined_features' column
    """
    df["combined_features"] = df.apply(combine_features, axis=1)
    print_step("Created 'combined_features' column (genres + keywords + cast + director + overview).")
    return df


def vectorize_text(df: pd.DataFrame) -> TfidfVectorizer:
    """
    Convert the 'combined_features' text column into TF-IDF vectors.

    TF-IDF (Term Frequency - Inverse Document Frequency) turns each
    movie's combined text into a numeric vector, where words that are
    common across ALL movies (like "the", "movie") get a lower weight,
    and rarer, more distinctive words get a higher weight.

    Parameters:
        df (pd.DataFrame): dataset with a 'combined_features' column

    Returns:
        scipy.sparse matrix: TF-IDF feature matrix (one row per movie)
    """
    tfidf = TfidfVectorizer(stop_words="english")
    feature_matrix = tfidf.fit_transform(df["combined_features"])
    print_step(f"Converted text into TF-IDF vectors with shape: {feature_matrix.shape}")
    return feature_matrix


def compute_similarity(feature_matrix) -> "np.ndarray":
    """
    Compute the Cosine Similarity between every pair of movies, based on
    their TF-IDF vectors.

    Cosine Similarity measures how "close" two vectors point in the same
    direction, giving a score between 0 (completely different) and 1
    (identical). This is a very common and beginner-friendly way to
    measure text similarity.

    Parameters:
        feature_matrix: TF-IDF matrix returned by vectorize_text()

    Returns:
        np.ndarray: a square matrix of similarity scores (movies x movies)
    """
    similarity_matrix = cosine_similarity(feature_matrix)
    print_step(f"Computed cosine similarity matrix with shape: {similarity_matrix.shape}")
    return similarity_matrix


def run_preprocessing_pipeline():
    """
    Run the full preprocessing pipeline, step by step, and save the
    final processed DataFrame + similarity matrix to the models/ folder.
    """
    print_step("Starting preprocessing pipeline...")

    df = load_dataset(DATASET_PATH)
    df = select_useful_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = create_combined_features(df)

    feature_matrix = vectorize_text(df)
    similarity_matrix = compute_similarity(feature_matrix)

    # We only need 'combined_features' for building the model, not for
    # displaying results later, but we keep it in case it's useful for
    # debugging. It does not add much size to the pickle file.
    save_pickle(df, MOVIES_PKL_PATH)
    save_pickle(similarity_matrix, SIMILARITY_PKL_PATH)

    print_step(f"Saved processed movie data to '{MOVIES_PKL_PATH}'.")
    print_step(f"Saved similarity matrix to '{SIMILARITY_PKL_PATH}'.")
    print_step("Preprocessing complete! You can now run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    run_preprocessing_pipeline()
