"""
Debug script to test TF-IDF calculation
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv('movies.csv')
print(f"Loaded {len(df)} movies")

# Check descriptions
print("\nSample descriptions:")
for i in range(3):
    print(f"{df.iloc[i]['title']}: {df.iloc[i]['description'][:100]}...")

# Create TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['description'])

print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Non-zero elements: {tfidf_matrix.nnz}")

# Calculate similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Cosine similarity matrix shape: {cosine_sim.shape}")

# Check similarity for The Dark Knight (index 2)
dark_knight_idx = 2
similarities = cosine_sim[dark_knight_idx]
print(f"\nSimilarities for The Dark Knight:")
for i, sim in enumerate(similarities[:10]):
    print(f"{df.iloc[i]['title']}: {sim:.3f}")