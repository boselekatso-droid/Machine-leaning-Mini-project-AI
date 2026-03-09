"""
Movie Recommendation System using TF-IDF and Cosine Similarity
This module contains the core machine learning logic for movie recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieRecommendationSystem:
    """
    A movie recommendation system that uses TF-IDF vectorization and cosine similarity
    to find similar movies based on their descriptions.
    """
    
    def __init__(self, csv_file='movies.csv'):
        """
        Initialize the recommendation system with movie data.
        
        Args:
            csv_file (str): Path to the CSV file containing movie data
        """
        self.csv_file = csv_file
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.cosine_sim = None
        
        # Load and prepare the data
        self.load_data()
        self.prepare_features()
        
    def load_data(self):
        """
        Load movie data from CSV file and perform basic preprocessing.
        """
        try:
            # Get the directory of the current script to find the CSV file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, self.csv_file)
            
            # Load the movie dataset
            self.movies_df = pd.read_csv(csv_path)
            
            # Basic data cleaning
            self.movies_df['description'] = self.movies_df['description'].fillna('')
            self.movies_df['title'] = self.movies_df['title'].fillna('')
            
            print(f"Loaded {len(self.movies_df)} movies from dataset")
            
        except FileNotFoundError:
            print(f"Error: Could not find {self.csv_file}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_features(self):
        """
        Prepare TF-IDF features from movie descriptions and calculate cosine similarity matrix.
        
        TF-IDF (Term Frequency-Inverse Document Frequency) converts text descriptions
        into numerical feature vectors that represent the importance of words.
        """
        # Combine description and genre for better similarity matching
        # This helps movies with similar genres to have higher similarity scores
        self.movies_df['combined_features'] = (
            self.movies_df['description'] + ' ' + 
            self.movies_df['genre'] + ' ' + 
            self.movies_df['genre']  # Add genre twice for more weight
        )
        
        # Initialize TF-IDF Vectorizer with adjusted parameters for better similarity detection
        # - stop_words='english': Remove common English words (the, and, is, etc.)
        # - max_features=1000: Use fewer features for better similarity detection with small dataset
        # - ngram_range=(1,2): Use both single words and two-word phrases
        # - min_df=1: Include words that appear in at least 1 document
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1
        )
        
        # Transform movie combined features into TF-IDF feature vectors
        # Each movie description becomes a vector of numbers representing word importance
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['combined_features'])
        
        # Calculate cosine similarity between all pairs of movies
        # Cosine similarity measures the angle between two vectors (0 = completely different, 1 = identical)
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print("TF-IDF features prepared and cosine similarity matrix calculated")
    
    def get_movie_index(self, movie_title):
        """
        Get the index of a movie in the dataset by its title.
        
        Args:
            movie_title (str): Title of the movie to find
            
        Returns:
            int: Index of the movie in the dataset, or None if not found
        """
        # Case-insensitive search for movie title
        movie_indices = self.movies_df[
            self.movies_df['title'].str.lower() == movie_title.lower()
        ].index
        
        if len(movie_indices) > 0:
            return movie_indices[0]
        return None
    
    def recommend(self, movie_title, num_recommendations=5):
        """
        Get movie recommendations based on a given movie title.
        
        Args:
            movie_title (str): Title of the movie to base recommendations on
            num_recommendations (int): Number of recommendations to return (default: 5)
            
        Returns:
            list: List of dictionaries containing recommended movie information
        """
        # Find the index of the input movie
        movie_idx = self.get_movie_index(movie_title)
        
        if movie_idx is None:
            return {"error": f"Movie '{movie_title}' not found in the dataset"}
        
        # Get similarity scores for all movies compared to the input movie
        sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
        
        # Sort movies by similarity score (highest first)
        # Skip the first one as it's the movie itself (similarity = 1.0)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
        
        # Get the top N most similar movies
        top_movies = sim_scores[:num_recommendations]
        
        # Prepare recommendations with movie details
        recommendations = []
        for idx, similarity_score in top_movies:
            movie_info = {
                'title': self.movies_df.iloc[idx]['title'],
                'genre': self.movies_df.iloc[idx]['genre'],
                'year': self.movies_df.iloc[idx]['year'],
                'rating': self.movies_df.iloc[idx]['rating'],
                'description': self.movies_df.iloc[idx]['description'],
                'similarity_score': round(similarity_score, 3)
            }
            recommendations.append(movie_info)
        
        return {
            'input_movie': movie_title,
            'recommendations': recommendations
        }
    
    def get_all_movie_titles(self):
        """
        Get a list of all movie titles in the dataset.
        
        Returns:
            list: List of all movie titles
        """
        return sorted(self.movies_df['title'].tolist())
    
    def search_movies(self, query):
        """
        Search for movies by partial title match.
        
        Args:
            query (str): Search query
            
        Returns:
            list: List of matching movie titles
        """
        if not query:
            return self.get_all_movie_titles()
        
        # Case-insensitive partial matching
        matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(query.lower(), na=False)
        ]['title'].tolist()
        
        return sorted(matches)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the recommendation system
    print("Initializing Movie Recommendation System...")
    recommender = MovieRecommendationSystem()
    
    # Test with a sample movie
    test_movie = "The Dark Knight"
    print(f"\nGetting recommendations for: {test_movie}")
    
    result = recommender.recommend(test_movie)
    
    if 'error' in result:
        print(result['error'])
    else:
        print(f"\nTop 5 movies similar to '{result['input_movie']}':")
        print("-" * 60)
        
        for i, movie in enumerate(result['recommendations'], 1):
            print(f"{i}. {movie['title']} ({movie['year']})")
            print(f"   Genre: {movie['genre']} | Rating: {movie['rating']}")
            print(f"   Similarity: {movie['similarity_score']}")
            print(f"   Description: {movie['description'][:100]}...")
            print()