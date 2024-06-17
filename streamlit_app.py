import os
import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from laserembeddings import Laser
import torch

# Set up LASER environment
LASER_DIR = "/path/to/your/laser/models"  # Change this to your path
os.environ["LASER"] = LASER_DIR

# Download LASER models if they are not present
if not os.path.exists(LASER_DIR):
    os.makedirs(LASER_DIR)
    os.system(f"wget -P {LASER_DIR} https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt")
    os.system(f"wget -P {LASER_DIR} https://dl.fbaipublicfiles.com/laser/models/bpe.codes.93langs")
    os.system(f"wget -P {LASER_DIR} https://dl.fbaipublicfiles.com/laser/models/eparl7.fnames")

# Initialize LASER
laser = Laser()

# Initialize ASR model and processor
ASR_MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME)

# Streamlit app interface
st.title('Semantic Similarity Analysis App')

st.header("Upload Source Text")
source_text = st.text_area("Enter the source text:")

st.header("Upload Target Text")
target_text = st.text_area("Enter the target text:")

st.header("Upload Audio for Target Text")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if st.button("Analyze"):
    if uploaded_file is not None:
        # Load audio file
        waveform, sample_rate = torchaudio.load(uploaded_file)
        
        # Process audio for ASR
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Decode the audio to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        target_text = transcription[0]
    
    # Calculate LASER embeddings
    source_embedding = laser.embed_sentences(source_text, lang='en')
    target_embedding = laser.embed_sentences(target_text, lang='en')
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(source_embedding), torch.tensor(target_embedding)
    )
    
    # Display results
    st.subheader("Target Text Transcription")
    st.write(target_text)
    
    st.subheader("Semantic Similarity Score")
    st.write(f"Cosine Similarity: {similarity.item()}")

    # Display most frequently repeated words in the target text
    from collections import Counter
    target_words = target_text.split()
    word_freq = Counter(target_words)
    most_common_words = word_freq.most_common(10)
    
    st.subheader("Most Frequently Repeated Words")
    st.write(most_common_words)
    
    # Detect filler words and long pauses in the target text (if uploaded as audio)
    filler_words = ['um', 'uh', 'eh', 'ah', 'like', 'you know']
    filler_count = {word: target_text.lower().count(word) for word in filler_words}
    
    st.subheader("Filler Words Count")
    st.write(filler_count)
    
    # Display long pauses
    if uploaded_file is not None:
        import numpy as np
        # Detect pauses longer than 3 seconds
        pause_threshold = 3 * sample_rate
        pauses = []
        current_pause = 0
        for i in range(waveform.shape[1]):
            if waveform[0, i] == 0:
                current_pause += 1
                if current_pause >= pause_threshold:
                    pauses.append(current_pause / sample_rate)
            else:
                current_pause = 0
        
        long_pauses = [pause for pause in pauses if pause >= 3]
        
        st.subheader("Long Pauses (in seconds)")
        st.write(long_pauses)
