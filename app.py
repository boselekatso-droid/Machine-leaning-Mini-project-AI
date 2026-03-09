"""
Streamlit Web Application for Movie Recommendation System
This app provides a simple and clean interface for getting movie recommendations.
"""

import streamlit as st
import pandas as pd
from recommendation import MovieRecommendationSystem
import os

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin-bottom: 1rem;
    }
    .similarity-score {
        background-color: #e3f2fd;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommendation_system():
    """
    Load the recommendation system and cache it for better performance.
    This function runs only once and caches the result.
    """
    try:
        return MovieRecommendationSystem()
    except Exception as e:
        st.error(f"Error loading recommendation system: {e}")
        return None

def display_movie_card(movie, rank):
    """
    Display a movie recommendation in a card format.
    
    Args:
        movie (dict): Movie information dictionary
        rank (int): Ranking position of the recommendation
    """
    with st.container():
        # Use simple Streamlit components instead of custom HTML
        st.subheader(f"#{rank} {movie['title']} ({movie['year']})")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Genre:** {movie['genre']}")
        with col2:
            st.write(f"**Rating:** ⭐ {movie['rating']}/10")
        with col3:
            st.write(f"**Similarity:** {movie['similarity_score']}")
        
        st.write(f"**Description:** {movie['description']}")
        st.divider()  # Add a separator line

def main():
    """
    Main application function that creates the Streamlit interface.
    """
    # App header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
    Discover movies similar to your favorites using AI-powered recommendations
    </p>
    """, unsafe_allow_html=True)
    
    # Load the recommendation system
    recommender = load_recommendation_system()
    
    if recommender is None:
        st.error("Failed to load the recommendation system. Please check if movies.csv exists.")
        return
    
    # Create two columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Movie selection interface
        st.subheader("🔍 Select a Movie")
        
        # Get all movie titles for the selectbox
        all_movies = recommender.get_all_movie_titles()
        
        # Search functionality
        search_query = st.text_input(
            "Search for a movie:",
            placeholder="Type to search for movies...",
            help="Start typing to filter the movie list"
        )
        
        # Filter movies based on search query
        if search_query:
            filtered_movies = recommender.search_movies(search_query)
            if not filtered_movies:
                st.warning("No movies found matching your search.")
                return
        else:
            filtered_movies = all_movies
        
        # Movie selection dropdown
        selected_movie = st.selectbox(
            "Choose a movie:",
            options=filtered_movies,
            help="Select a movie to get recommendations based on it"
        )
    
    with col2:
        st.subheader("📊 Settings")
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=3,
            max_value=10,
            value=5,
            help="Choose how many movie recommendations you want"
        )
    
    # Generate recommendations button
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        if selected_movie:
            with st.spinner("🤖 Finding similar movies..."):
                # Get recommendations
                result = recommender.recommend(selected_movie, num_recommendations)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    # Display selected movie info
                    st.success(f"✅ Recommendations based on: **{result['input_movie']}**")
                    
                    # Display recommendations
                    st.subheader("🎬 Recommended Movies")
                    
                    if result.get('recommendations'):
                        for i, movie in enumerate(result['recommendations'], 1):
                            display_movie_card(movie, i)
                    else:
                        st.info("No recommendations found for this movie.")
        else:
            st.warning("Please select a movie first!")
    
    # Information section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### About this Recommendation System
        
        This movie recommendation system uses **Machine Learning** to find movies similar to your selection:
        
        1. **TF-IDF Vectorization**: Converts movie descriptions into numerical features that represent the importance of words
        2. **Cosine Similarity**: Calculates how similar movies are based on their description vectors
        3. **Ranking**: Returns the top movies with highest similarity scores
        
        ### Features
        - 🎯 **Accurate recommendations** based on movie content
        - 🔍 **Smart search** with partial matching
        - 📱 **Responsive design** that works on all devices
        - ⚡ **Fast performance** with cached computations
        
        ### Dataset
        The system uses a curated dataset of popular movies with descriptions, genres, ratings, and release years.
        """)
    
    # Footer
    st.markdown("""
    ---
    <p style="text-align: center; color: #999; font-size: 0.9rem;">
    Built with ❤️ using Streamlit and scikit-learn | Movie Recommendation System v1.0
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()