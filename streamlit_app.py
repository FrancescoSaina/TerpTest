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

# Function to compute similarity matrix and tokens
def compute_similarity_matrix(source_text, target_text):
    # Tokenize texts
    source_tokens = tokenizer.tokenize(source_text)
    target_tokens = tokenizer.tokenize(target_text)
    
    # Encode tokens
    inputs = tokenizer(source_text, target_text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        # Get embeddings from BERT
        embeddings = model(**inputs).last_hidden_state
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.cosine_similarity(embeddings[0], embeddings[1], dim=1).reshape(len(source_tokens), len(target_tokens))
        
    return similarity_matrix, source_tokens, target_tokens

# Function to plot similarity heatmap
def plot_similarity_heatmap(similarity_matrix, source_tokens, target_tokens):
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='YlGnBu', interpolation='nearest')
    plt.colorbar()
    
    # Set axis labels and ticks
    plt.xticks(ticks=np.arange(len(target_tokens)), labels=target_tokens, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(source_tokens)), labels=source_tokens)
    
    # Set plot title and labels
    plt.title('Semantic Similarity Heat Map')
    plt.xlabel('Target Tokens')
    plt.ylabel('Source Tokens')
    
    # Show plot
    st.pyplot(plt)

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
        
        # Compute and display heat map
        if source_text and transcript:
            similarity_matrix, source_tokens, target_tokens = compute_similarity_matrix(source_text, transcript)
            st.subheader("Semantic Similarity Heat Map")
            plot_similarity_heatmap(similarity_matrix.cpu().numpy(), source_tokens, target_tokens)

if __name__ == "__main__":
    main()
