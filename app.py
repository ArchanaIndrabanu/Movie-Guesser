import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load dataset
df = pd.read_csv('wiki_movie_plots_deduped.csv')

# Take a subset of the dataset from rows 31,495 to 28,897 (in reverse order)
subset_df = df.iloc[28897:31495]  # Corrected to select rows in ascending order for proper operation

# Initialize the model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Encode the movie plots
plots = subset_df['Plot'].tolist()
plot_embeddings = model.encode(plots, convert_to_tensor=True)

# Define the movie finding function
def find_movie(description):
    # Encode the description
    description_embedding = model.encode(description, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(description_embedding, plot_embeddings)[0]
    
    # Find the index of the highest cosine similarity
    top_result = torch.argmax(cosine_scores).item()
    
    # Return the title of the most similar movie
    return subset_df.iloc[top_result]['Title']

# Streamlit App
st.title('Movie Guessing App')

description = st.text_input('Enter a movie description:')

if description:
    movie_title = find_movie(description)
    st.write(f'The most similar movie is: {movie_title}')
