import streamlit as st
import torch
import torchaudio
from transformers import pipeline
from laserembeddings import Laser

# Verify package versions
st.write("Torch version:", torch.__version__)
st.write("Torchaudio version:", torchaudio.__version__)

# Load a pipeline
st.write("Loading sentiment analysis pipeline...")
sentiment_pipeline = pipeline("sentiment-analysis")

st.write("Sentiment analysis pipeline loaded.")

def analyze_sentiment(text):
    return sentiment_pipeline(text)

st.write("Loading LASER model...")
laser = Laser()
st.write("LASER model loaded.")

def embed_sentence(sentence):
    return laser.embed_sentences([sentence], lang='en')

st.title("Sentiment Analysis and Sentence Embedding")

input_text = st.text_input("Enter text for sentiment analysis:")
if input_text:
    sentiment = analyze_sentiment(input_text)
    st.write("Sentiment Analysis Result:", sentiment)

input_sentence = st.text_input("Enter sentence for embedding:")
if input_sentence:
    embedding = embed_sentence(input_sentence)
    st.write("Sentence Embedding Result:", embedding)
