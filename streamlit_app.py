import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

def main():
    st.title("Semantic Similarity Analysis")

    # File uploader for source text
    st.subheader("Upload Source Text")
    source_text = st.file_uploader("Upload a text file or provide a direct link")

    # File uploader for target text
    st.subheader("Upload Target Text")
    target_text = st.file_uploader("Upload a text file or provide a direct link")

    # Semantic similarity calculation
    if source_text and target_text:
        st.subheader("Semantic Similarity Score")

        # Load a pre-trained Sentence Transformer model
        model = SentenceTransformer('distiluse-base-multilingual-cased')

        # Encode sentences
        source_embeddings = model.encode(source_text.read().decode('utf-8').splitlines())
        target_embeddings = model.encode(target_text.read().decode('utf-8').splitlines())

        # Calculate similarity scores
        similarity_scores = util.pytorch_cos_sim(source_embeddings, target_embeddings)

        # Display similarity score
        st.write(f"Similarity Score: {similarity_scores.item()}")

if __name__ == "__main__":
    main()
