import streamlit as st 
import pandas as pd 
import torch 
from sentence_transformers import SentenceTransformer, util 
ECHO is on.
"# Load dataset" 
df = pd.read_csv('//content//wiki_movie_plots_deduped (1).csv') 
ECHO is on.
"# Take a subset of the dataset from rows 31,495 to 28,897 (in reverse order)" 
subset_df = df.iloc[31495:28896:-1] 
ECHO is on.
"# Initialize the model" 
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2') 
ECHO is on.
"# Encode the movie plots" 
plots = subset_df['Plot'].tolist() 
plot_embeddings = model.encode(plots, convert_to_tensor=True) 
ECHO is on.
"def find_movie(description):" 
"    description_embedding = model.encode(description, convert_to_tensor=True)" 
"    cosine_scores = util.pytorch_cos_sim(description_embedding, plot_embeddings)[0]" 
"    top_result = torch.argmax(cosine_scores).item()" 
"    return subset_df.iloc[top_result]['Title']" 
ECHO is on.
"# Streamlit App" 
"st.title('Movie Guessing App')" 
"description = st.text_input('Enter a movie description:')" 
"if description:" 
"    movie_title = find_movie(description)" 
"    st.write(f'The most similar movie is: {movie_title}')" 
