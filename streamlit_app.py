import streamlit as st
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title of the web app
st.title("Speech-to-Text with Transformers")

# File uploader for audio files
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

# Load the speech recognition model
@st.cache_resource
def load_asr_model():
    asr_model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    return asr_model

# Load the semantic similarity model
@st.cache_resource
def load_similarity_model():
    model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
    return model, tokenizer

asr_model = load_asr_model()
similarity_model, similarity_tokenizer = load_similarity_model()

if uploaded_file is not None:
    # Display file details
    st.audio(uploaded_file, format=uploaded_file.type)
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"File type: {uploaded_file.type}")

    # Load audio file
    waveform, sample_rate = torchaudio.load(uploaded_file)
    waveform = waveform.numpy()

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(torch.Tensor(waveform)).numpy()
        sample_rate = 16000

    # Perform speech-to-text
    st.write("Transcribing audio...")
    transcription = asr_model(waveform, sampling_rate=sample_rate)
    st.write("Transcription:", transcription['text'])

    # Semantic similarity check
    reference_text = st.text_area("Enter reference text for semantic similarity check:", "")
    if reference_text:
        st.write("Calculating semantic similarity...")
        inputs = similarity_tokenizer([reference_text, transcription['text']], return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = similarity_model(**inputs)
            similarity_score = cosine_similarity(outputs.logits[0].unsqueeze(0), outputs.logits[1].unsqueeze(0))[0][0]
        st.write(f"Semantic similarity score: {similarity_score:.2f}")

    # Word frequency analysis
    st.write("Calculating word frequencies...")
    vectorizer = CountVectorizer()
    word_counts = vectorizer.fit_transform([transcription['text']]).toarray().flatten()
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts))
    word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    st.write(word_freq_df)

    # Filler word analysis
    st.write("Analyzing filler words...")
    fillers = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'seriously', 'literally']
    filler_counts = {filler: transcription['text'].lower().split().count(filler) for filler in fillers}
    filler_counts_df = pd.DataFrame(list(filler_counts.items()), columns=['Filler Word', 'Count'])
    st.write(filler_counts_df)
else:
    st.write("Please upload an audio file to transcribe.")
