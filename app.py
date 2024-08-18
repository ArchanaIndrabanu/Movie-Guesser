import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Page Config
st.set_page_config(page_title="Movie Guessing App", page_icon="🎥", layout="centered")

# Custom CSS to increase font size
st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
    }
    .medium-font {
        font-size: 20px !important;
    }
    .small-font {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Header
st.markdown('<p class="big-font">🎬 Movie Guessing App</p>', unsafe_allow_html=True)
st.markdown("""
    <p class="medium-font">
    **Welcome to the Movie Guessing App!**  
    Enter a brief description of a movie, and we'll guess the movie title for you.  
    This app uses advanced natural language processing to match your description with movie plots.
    </p>
    """, unsafe_allow_html=True)

st.divider()

# Load dataset
df = pd.read_csv('wiki_movie_plots_deduped.csv')

# Take a subset of the dataset from rows 31,495 to 28,897 (in reverse order)
subset_df = df.iloc[28897:31495]  # Selecting rows in ascending order

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

# Streamlit App Input and Output
st.markdown('<p class="medium-font">Find Your Movie 🎥</p>', unsafe_allow_html=True)
description = st.text_input('Enter a movie description:', help="Describe a movie plot here.")

if description:
    st.markdown('<p class="medium-font">🔍 Searching for the most similar movie...</p>', unsafe_allow_html=True)
    movie_title = find_movie(description)
    st.markdown(f'<p class="medium-font">**The most similar movie is: _{movie_title}_**</p>', unsafe_allow_html=True)

    # Optionally, display more details about the movie
    movie_details = subset_df[subset_df['Title'] == movie_title].iloc[0]
    st.markdown(f"<p class='small-font'><strong>Plot:</strong> {movie_details['Plot']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='small-font'><strong>Release Year:</strong> {movie_details['Release Year']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='small-font'><strong>Genre:</strong> {movie_details['Genre']}</p>", unsafe_allow_html=True)
else:
    st.markdown('<p class="medium-font">Please enter a movie description to start the search.</p>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
    <p class="small-font">
    **App Developed by [Your Name]**  
    _Powered by Streamlit and Sentence Transformers_
    </p>
    """, unsafe_allow_html=True)
