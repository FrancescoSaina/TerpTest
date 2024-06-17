import streamlit as st
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel, pipeline
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Initialize DeepSpeech for ASR
model_path = "path_to_deepspeech_model"  # Replace with your DeepSpeech model path
model = torch.hub.load('mozilla/DeepSpeech', 'deepspeech:v0.9.3', model_path=model_path)
model.eval()

# Initialize Hugging Face Transformer model
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_transformer = AutoModel.from_pretrained(model_name)
similarity_pipeline = pipeline('feature-extraction', model=model_transformer, tokenizer=tokenizer)

def deepspeech_asr(audio_file):
    waveform, sample_rate = librosa.load(audio_file, sr=None)
    waveform_tensor = torch.tensor(waveform)
    text = model.stt(waveform_tensor)
    return text

def semantic_similarity(source_text, target_text):
    source_encoding = similarity_pipeline(source_text, return_tensors="pt")
    target_encoding = similarity_pipeline(target_text, return_tensors="pt")
    similarity_score = torch.nn.functional.cosine_similarity(source_encoding, target_encoding).item()
    return similarity_score

def plot_heatmap(audio_file):
    waveform, sample_rate = librosa.load(audio_file, sr=None)
    f, t, Sxx = signal.spectrogram(waveform, sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    st.pyplot()

def main():
    st.title("Automatic Speech Recognition and Semantic Similarity Analysis")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Perform ASR using DeepSpeech
        st.subheader("Speech Recognition Result")
        text_result = deepspeech_asr(uploaded_file)
        st.write(f"Recognized Text: {text_result}")

        # Semantic similarity with a reference text
        st.subheader("Semantic Similarity Analysis")
        reference_text = st.text_area("Enter reference text")
        if st.button("Calculate Similarity"):
            if reference_text:
                similarity_score = semantic_similarity(reference_text, text_result)
                st.write(f"Semantic Similarity Score: {similarity_score:.4f}")

        # Display spectrogram heatmap
        st.subheader("Spectrogram")
        if st.button("Show Spectrogram"):
            plot_heatmap(uploaded_file)

if __name__ == "__main__":
    main()
