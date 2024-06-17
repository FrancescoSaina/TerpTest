import streamlit as st
import torch
import torchaudio
from transformers import BertModel, BertTokenizer
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import tempfile
import os
import deepspeech

# Load the BERT model and tokenizer (example with mBERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Load DeepSpeech model for ASR (change path as necessary)
MODEL_PATH = 'path/to/deepspeech-0.9.3-models.pbmm'
SCORER_PATH = 'path/to/deepspeech-0.9.3-models.scorer'
ds = deepspeech.Model(MODEL_PATH)
ds.enableExternalScorer(SCORER_PATH)

# Function to perform ASR using DeepSpeech
def perform_asr(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name
    # Perform ASR using DeepSpeech
    audio, rate = librosa.load(temp_audio_path, sr=16000)
    transcript = ds.stt(audio)
    os.remove(temp_audio_path)
    return transcript

# Function to compute semantic similarity using BERT embeddings
def compute_semantic_similarity(source_text, target_text):
    # Tokenize and encode the texts
    inputs = tokenizer(source_text, target_text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        # Get embeddings from BERT
        embeddings = model(**inputs).last_hidden_state
        # Compute cosine similarity between embeddings
        similarity_score = torch.cosine_similarity(embeddings[0], embeddings[1], dim=1)
    return similarity_score.item()

# Function to analyze text characteristics (repeated words, filler words, pauses)
def analyze_text_characteristics(target_text):
    # Split text into words
    words = target_text.split()
    # Compute word frequencies
    word_freq = {word: words.count(word) for word in set(words)}
    # Get most frequent words
    most_frequent_words = sorted(word_freq, key=word_freq.get, reverse=True)[:5]
    
    # Find filler words (example list, can be expanded)
    filler_words = ['ehm', 'uhm']
    fillers = [word for word in words if word.lower() in filler_words]
    
    # Compute pauses longer than 3 seconds using librosa
    pauses = librosa.effects.split(audio, top_db=20)
    long_pauses = [((end-start)/rate) for start, end in pauses if (end-start)/rate > 3]

    return most_frequent_words, fillers, long_pauses

# Streamlit app
def main():
    st.title("Interpreting Feedback Tool")
    
    st.header("Upload Reference Source Text")
    source_text = st.text_area("Paste or upload the source text here")
    
    st.header("Upload Target Rendition Text or Audio")
    target_text_file = st.file_uploader("Upload target text file (or audio file)", type=["txt", "mp3", "wav"])
    
    # Convert audio to text using DeepSpeech ASR
    if target_text_file is not None:
        if target_text_file.type == "audio/wav" or target_text_file.type == "audio/mp3":
            transcript = perform_asr(target_text_file)
        else:
            transcript = str(target_text_file.read(), "utf-8")
    
        st.subheader("Automatic Speech Recognition (ASR) Transcript")
        st.write(transcript)
    
        # Compute semantic similarity
        if source_text:
            similarity_score = compute_semantic_similarity(source_text, transcript)
            st.subheader("Semantic Similarity Score")
            st.write(f"Similarity score: {similarity_score:.4f}")
        
        # Analyze text characteristics
        if transcript:
            most_frequent_words, fillers, pauses = analyze_text_characteristics(transcript)
            st.subheader("Most Frequently Repeated Words")
            st.write(", ".join(most_frequent_words))
            st.subheader("Filler Words (e.g., 'ehm', 'uhm')")
            st.write(", ".join(fillers))
            st.subheader("Pauses Longer Than 3 Seconds")
            st.write(", ".join(map(str, pauses)))

if __name__ == "__main__":
    main()
