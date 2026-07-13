"""
app.py
------
This is the main Streamlit web application for the Movie Recommendation
System. It provides a simple, user-friendly interface where a user can:

- Pick (or type) a movie they like
- Click "Recommend" to see the Top 10 most similar movies
- Try a "Surprise Me" random recommendation
- See their recently searched movies and recommendation history
- View basic statistics about the dataset

Run this app with:
    streamlit run app.py
"""

import random

import streamlit as st

from recommendation import get_all_titles, get_recommendations, load_model_data
from utils import print_step

# ------------------------------------------------------------------ #
# PAGE CONFIGURATION
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="CineMatch | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ #
# CACHED DATA LOADING
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner=False)
def get_data():
    """
    Load the processed movie data and similarity matrix ONCE and cache
    it, so it isn't reloaded from disk on every single user interaction.
    Streamlit re-runs the whole script on every click, so caching here
    is essential for a responsive app.
    """
    movies_df, similarity_matrix = load_model_data()
    all_titles = get_all_titles(movies_df)
    return movies_df, similarity_matrix, all_titles


# Try to load the model files. If preprocess.py hasn't been run yet,
# show a friendly error instead of letting the app crash.
try:
    movies_df, similarity_matrix, all_titles = get_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# ------------------------------------------------------------------ #
# SESSION STATE (keeps track of history during the current session)
# ------------------------------------------------------------------ #
if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = []          # list of movie titles searched

if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = []    # list of (movie, [recommended titles])

if "last_result" not in st.session_state:
    st.session_state.last_result = None             # holds the last recommendation result


def add_to_history(searched_movie: str, recommended_titles: list) -> None:
    """
    Update the session's recent-searches list and recommendation history.
    Keeps only the last 5 searches to keep the sidebar clean.
    """
    if searched_movie not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, searched_movie)
        st.session_state.recent_searches = st.session_state.recent_searches[:5]

    st.session_state.recommendation_history.insert(0, (searched_movie, recommended_titles))
    st.session_state.recommendation_history = st.session_state.recommendation_history[:5]


def reset_session():
    """Clear all session history and the last shown result."""
    st.session_state.recent_searches = []
    st.session_state.recommendation_history = []
    st.session_state.last_result = None


# ------------------------------------------------------------------ #
# SIDEBAR
# ------------------------------------------------------------------ #
with st.sidebar:
    st.title("🎬 CineMatch")
    st.caption("A Content-Based Movie Recommendation System")

    st.markdown("---")
    st.subheader("ℹ️ About This Project")
    st.write(
        "This app recommends movies similar to one you already like, "
        "using **Content-Based Filtering**. It analyzes each movie's "
        "genres, keywords, overview, cast and director, then finds "
        "movies with the most similar content using **TF-IDF** and "
        "**Cosine Similarity**."
    )

    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    if data_loaded:
        st.metric("Total Movies", len(movies_df))
        st.metric("Unique Directors", movies_df["director"].nunique())
    else:
        st.warning("Dataset not loaded yet.")

    st.markdown("---")
    st.subheader("🕘 Recently Searched")
    if st.session_state.recent_searches:
        for movie in st.session_state.recent_searches:
            st.write(f"• {movie}")
    else:
        st.caption("No searches yet in this session.")

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("---")

# ------------------------------------------------------------------ #
# MAIN PAGE
# ------------------------------------------------------------------ #
st.title("🎬 CineMatch — Find Your Next Favorite Movie")
st.write(
    "Select a movie you enjoyed, and we'll recommend **10 similar movies** "
    "based on genre, story, cast and director."
)

if not data_loaded:
    st.error(
        "⚠️ Could not find the trained model files (`models/movies_data.pkl` "
        "and `models/similarity.pkl`).\n\n"
        "Please run the preprocessing script first from your terminal:\n\n"
        "`python preprocess.py`"
    )
    st.stop()  # Stop the app here since there is nothing else we can do

st.markdown("### 🔍 Choose a Movie")

col1, col2 = st.columns([3, 1])

with col1:
    # st.selectbox naturally supports "type to search" filtering,
    # which acts as our search-suggestions feature.
    selected_movie = st.selectbox(
        "Start typing a movie name...",
        options=[""] + all_titles,
        index=0,
        help="Start typing to filter the list. Small spelling mistakes are okay too!",
    )

with col2:
    st.write("")  # spacing to align button with selectbox
    st.write("")
    surprise_me = st.button("🎲 Surprise Me!", use_container_width=True)

# Also allow free-text typing, in case the user's movie isn't in the dropdown
# or they prefer typing (this is where fuzzy/typo matching really helps).
typed_movie = st.text_input(
    "...or type a movie name manually (typos are okay!)",
    placeholder="e.g. Avngers, Titanik, Jurasic Park",
)

recommend_clicked = st.button("🎯 Recommend Movies", type="primary", use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------------ #
# HANDLE "SURPRISE ME" BUTTON
# ------------------------------------------------------------------ #
if surprise_me:
    random_movie = random.choice(all_titles)
    with st.spinner(f"Picking a random movie and finding similar ones..."):
        result = get_recommendations(random_movie, movies_df, similarity_matrix, top_n=10)
    st.session_state.last_result = result
    if result["success"]:
        add_to_history(
            result["matched_title"],
            [rec["title"] for rec in result["recommendations"]],
        )

# ------------------------------------------------------------------ #
# HANDLE "RECOMMEND MOVIES" BUTTON
# ------------------------------------------------------------------ #
if recommend_clicked:
    # Prefer the typed movie name if the user typed something,
    # otherwise fall back to the dropdown selection.
    movie_query = typed_movie.strip() if typed_movie.strip() else selected_movie

    if not movie_query:
        st.warning("⚠️ Please select or type a movie name before clicking Recommend.")
    else:
        with st.spinner(f"Finding movies similar to '{movie_query}'..."):
            result = get_recommendations(movie_query, movies_df, similarity_matrix, top_n=10)
        st.session_state.last_result = result

        if result["success"]:
            add_to_history(
                result["matched_title"],
                [rec["title"] for rec in result["recommendations"]],
            )


# ------------------------------------------------------------------ #
# DISPLAY RESULTS
# ------------------------------------------------------------------ #
result = st.session_state.last_result

if result is not None:
    if result["success"]:
        st.success(f"✅ {result['message']}")

        # Let the user know if we auto-corrected a typo
        st.markdown(f"### 🍿 Top {len(result['recommendations'])} Recommendations")

        for rec in result["recommendations"]:
            with st.expander(f"**#{rec['rank']} — {rec['title']}**  ({rec['similarity_score']}% match)"):
                left, right = st.columns([1, 2])
                with left:
                    st.write(f"**🎭 Genres:** {rec['genres'] or 'N/A'}")
                    st.write(f"**🎬 Director:** {rec['director'] or 'N/A'}")
                    st.write(f"**📅 Release Date:** {rec['release_date'] or 'N/A'}")
                    st.write(f"**⭐ Rating:** {rec['vote_average']}/10")
                with right:
                    st.write(f"**👥 Cast:** {rec['cast'] or 'N/A'}")
                    st.write(f"**📝 Overview:** {rec['overview'] or 'No overview available.'}")
    else:
        st.error(f"❌ {result['message']}")
else:
    st.info("👆 Select or type a movie above, then click **Recommend Movies** to get started.")


# ------------------------------------------------------------------ #
# RECOMMENDATION HISTORY (this session)
# ------------------------------------------------------------------ #
if st.session_state.recommendation_history:
    st.markdown("---")
    st.markdown("### 📜 Recommendation History (This Session)")
    for movie, recs in st.session_state.recommendation_history:
        st.write(f"**{movie}** → " + ", ".join(recs[:5]) + ("..." if len(recs) > 5 else ""))
