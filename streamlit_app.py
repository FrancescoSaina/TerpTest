import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load multilingual sentence embeddings model
model_name = 'sentence-transformers/LaBSE'
model = SentenceTransformer(model_name)

def main():
    st.title('Semantic Similarity Comparison')

    # Input text fields
    source_text = st.text_area('Enter source text (language 1)', '')
    target_text = st.text_area('Enter target text (language 2)', '')

    if st.button('Compare'):
        if source_text.strip() == '' or target_text.strip() == '':
            st.warning('Please enter both source and target texts.')
        else:
            # Encode texts into embeddings
            source_embedding = model.encode(source_text, convert_to_tensor=True)
            target_embedding = model.encode(target_text, convert_to_tensor=True)

            # Calculate cosine similarity
            similarity_score = util.pytorch_cos_sim(source_embedding, target_embedding).item()

            # Display similarity score
            st.success(f'Semantic Similarity Score: {similarity_score:.4f}')

if __name__ == '__main__':
    main()
