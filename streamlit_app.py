import streamlit as st
from transformers import BertTokenizer, BertModel, pipeline
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
bert_pipeline = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

def analyze_semantic_similarity(source_text, target_text):
    # Perform semantic similarity analysis using BERT embeddings
    source_embedding = bert_pipeline(source_text)[0]
    target_embedding = bert_pipeline(target_text)[0]

    # Calculate similarity score (cosine similarity)
    similarity_score = np.dot(source_embedding, target_embedding) / (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
    
    return similarity_score

def analyze_audio(audio_file):
    # Perform audio analysis
    waveform, sample_rate = librosa.load(audio_file, sr=None)
    durations = librosa.effects.split(waveform, top_db=20)
    pauses = np.array([duration[1] - duration[0] for duration in durations])

    # Calculate filler words and hesitations
    filler_words = ["ehm", "uhm"]  # Example filler words, you can expand this list
    filler_word_count = sum(waveform.lower().count(w) for w in filler_words)
    hesitation_count = len(pauses[pauses > 3])

    # Plot waveform and pauses
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(waveform)
    plt.title('Waveform')
    plt.subplot(2, 1, 2)
    plt.plot(gaussian_filter1d(pauses, sigma=2))
    plt.title('Pauses (smoothed)')
    st.pyplot()

    return filler_word_count, hesitation_count

def main():
    st.title('Semantic Similarity and Audio Analysis')

    st.header('Upload Source Text')
    source_text = st.text_area('Enter or upload source text')

    st.header('Upload Target Text/Audio')
    target_text = st.text_area('Enter target text or upload audio (.wav)')

    if st.button('Analyze'):
        similarity_score = analyze_semantic_similarity(source_text, target_text)

        if target_text.strip().endswith('.wav'):
            audio_file = target_text.strip()
            filler_words_count, hesitation_count = analyze_audio(audio_file)
            st.write(f'Filler Words Count: {filler_words_count}')
            st.write(f'Hesitation Count (>3s): {hesitation_count}')
        else:
            st.write(f'Semantic Similarity Score: {similarity_score}')

if __name__ == "__main__":
    main()
