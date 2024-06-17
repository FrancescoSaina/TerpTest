import streamlit as st
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import wavfile

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
    transcribed_text = transcription['text']
    st.write("Transcription:", transcribed_text)

    # Semantic similarity check
    reference_text = st.text_area("Enter reference text for semantic similarity check:", "")
    if reference_text:
        st.write("Calculating semantic similarity...")
        inputs = similarity_tokenizer([reference_text, transcribed_text], return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = similarity_model(**inputs)
            similarity_score = cosine_similarity(outputs.logits[0].unsqueeze(0), outputs.logits[1].unsqueeze(0))[0][0]
        st.write(f"Semantic similarity score: {similarity_score:.2f}")

    # Word frequency analysis
    st.write("Calculating word frequencies...")
    vectorizer = CountVectorizer()
    word_counts = vectorizer.fit_transform([transcribed_text]).toarray().flatten()
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts))
    word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    st.write(word_freq_df)

    # Frequently repeated words
    st.write("Frequently repeated words:")
    repeated_words_df = word_freq_df[word_freq_df['Frequency'] > 1].sort_values(by='Frequency', ascending=False)
    st.write(repeated_words_df)

    # Filler word analysis
    st.write("Analyzing filler words...")
    fillers = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'seriously', 'literally', 'ehm', 'uhm']
    filler_counts = {filler: transcribed_text.lower().split().count(filler) for filler in fillers}
    filler_counts_df = pd.DataFrame(list(filler_counts.items()), columns=['Filler Word', 'Count'])
    st.write(filler_counts_df)

    # Detecting pauses longer than 3 seconds
    st.write("Detecting pauses longer than 3 seconds...")
    def detect_pauses(waveform, sample_rate, threshold=3):
        energy = np.mean(waveform**2, axis=0)
        silence_threshold = 0.01 * np.max(energy)
        pause_durations = []
        current_pause_duration = 0
        for e in energy:
            if e < silence_threshold:
                current_pause_duration += 1
            else:
                if current_pause_duration >= threshold * sample_rate:
                    pause_durations.append(current_pause_duration / sample_rate)
                current_pause_duration = 0
        return pause_durations

    pauses = detect_pauses(waveform, sample_rate)
    if pauses:
        pause_times_df = pd.DataFrame(pauses, columns=['Pause Duration (seconds)'])
        st.write(pause_times_df)
    else:
        st.write("No pauses longer than 3 seconds detected.")

else:
    st.write("Please upload an audio file to transcribe.")
