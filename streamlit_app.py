import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

def main():
    st.title("Semantic Similarity Analysis")

    # File uploader for source text
    st.subheader("Upload Source Text")
    source_text = st.file_uploader("Upload a text file or provide a direct link")

    # Method to upload or input target text
    st.subheader("Upload or Input Target Text")
    target_option = st.radio("Choose method:", ("Upload Text File", "Input Text"))
    if target_option == "Upload Text File":
        target_text = st.file_uploader("Upload a text file or provide a direct link")
    else:
        target_text = st.text_area("Input the target text:")

    # Semantic similarity calculation
    if source_text and target_text:
        st.subheader("Semantic Similarity Score")

        # Load a pre-trained Sentence Transformer model
        model = SentenceTransformer('distiluse-base-multilingual-cased')

        # Encode sentences
        source_embeddings = model.encode(source_text.read().decode('utf-8').splitlines())
        target_embeddings = model.encode(target_text)

        # Calculate similarity scores
        similarity_scores = util.pytorch_cos_sim(source_embeddings, target_embeddings)

        # Display similarity score
        st.write(f"Similarity Score: {similarity_scores.item()}")

    st.sidebar.title("Additional Analysis (if target is audio)")
    st.sidebar.subheader("Top Repeated Words in Target Text")

    st.sidebar.subheader("Filler Words or Hesitations")
    st.sidebar.subheader("Pauses Longer Than 3 Seconds")

if __name__ == "__main__":
    main()
