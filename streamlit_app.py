import gradio as gr
from sentence_transformers import SentenceTransformer, util
import SpeechRecognition as sr
from pydub import AudioSegment
from collections import Counter

# Initialize models
model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# Function to calculate semantic similarity
def calculate_semantic_similarity(text1, text2):
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    similarity_score = util.cos_sim(embeddings1, embeddings2).item()
    return similarity_score

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_file)
    with audio as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# Function to analyze text features
def analyze_text(text):
    words = text.split()
    word_freq = Counter(words)
    filler_words = ['ehm', 'uhm']  # Customize as needed
    filler_count = sum(word_freq.get(word, 0) for word in filler_words)
    # Analyze pauses in audio (to be implemented)
    return word_freq, filler_count

# Define Gradio interface components
def semantic_similarity_interface(source_text, target_audio_or_text):
    if isinstance(target_audio_or_text, str):  # if target is text
        target_text = target_audio_or_text
    else:  # if target is audio
        target_audio = target_audio_or_text.name
        target_text = transcribe_audio(target_audio)

    similarity_score = calculate_semantic_similarity(source_text, target_text)
    word_freq, filler_count = analyze_text(target_text)

    return {
        'Semantic Similarity Score': similarity_score,
        'Most Frequent Words': dict(word_freq.most_common(5)),
        'Filler Words Count': filler_count
    }

# Create Gradio interface
iface = gr.Interface(
    fn=semantic_similarity_interface,
    inputs=[
        gr.Textbox(label="Enter or upload your source text"),
        gr.Audio(label="Upload an audio file or speak into microphone", type="file")
    ],
    outputs="json",
    live=True,
    title="Semantic Similarity and Text Analysis App"
)

# Launch the interface
iface.launch()
