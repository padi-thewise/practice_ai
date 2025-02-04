# -*- coding: utf-8 -*-
"""repo_rec.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1udW94xUIUidhnufnSB-gWegk2qU4TbLU
"""

import warnings
warnings.filterwarnings('ignore')


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

# Step 1: Load Dataset and Precompute Embeddings
@st.cache_resource
def load_data_and_model():
    """
    Load the dataset and precompute embeddings. Load the CodeBERT model and tokenizer.
    """
    # Load dataset
    dataset_path =  '/content/drive/MyDrive/practice_ml/filtered_dataset.parquet'
    data = pd.read_parquet(dataset_path)

    # Combine text fields for embedding generation
    data['text'] = data['docstring'].fillna('') + ' ' + data['summary'].fillna('')

    # Load t5-small model and tokenizer
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Precompute embeddings
    @st.cache_resource
    def generate_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    data['embedding'] = data['text'].apply(lambda x: generate_embedding(x))
    return data, tokenizer, model

# Load resources
data, tokenizer, model = load_data_and_model()

# Step 2: Define Functions
def generate_query_embedding(query):
    """
    Generate an embedding for the user's query.
    """
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_recommendations(query, top_n=5):
    """
    Process the user query, calculate similarity, and recommend repositories.
    """
    query_embedding = generate_query_embedding(query)
    data['similarity'] = data['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    recommendations = data.sort_values(by='similarity', ascending=False).head(top_n)
    return recommendations

# Step 3: Streamlit App Layout
st.title("Repo Recommender System")
st.markdown(
    """
    **Welcome to Repo_Recommender** Enter a brief description of your project to get repository recommendations.
    """
)

# User Input
user_query = st.text_area("Describe your project (e.g., tools, goals, etc.):", height=150)

# Get Recommendations
if st.button("Get Recommendations"):
    if user_query.strip():
        st.write("Processing your query...")
        recommendations = get_recommendations(user_query)

        # Display Recommendations
        st.markdown("### Top Recommendations:")
        for idx, row in recommendations.iterrows():
            st.markdown(
                f"""
                **{idx + 1}. Repository:** {row['repo']}
                **Path:** {row['path']}
                **Summary:** {row['summary']}
                **URL:** [Link]({row['url']})
                """
            )
    else:
        st.warning("Please enter a project description to get recommendations.")

from google.colab import drive
drive.mount('/content/drive')

