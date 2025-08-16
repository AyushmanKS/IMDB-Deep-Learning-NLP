import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --- Caching and Model Loading ---
@st.cache_resource
def load_model_and_processor():
    """Load the Keras model and word_index once."""
    model = load_model('simple_rnn_imdb.h5')
    word_index = imdb.get_word_index()
    return model, word_index

model, word_index = load_model_and_processor()

# --- Preprocessing Function ---
def preprocess_text(text, word_index, maxlen=500):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

# --- UI Layout ---
st.title('🎬 Movie Review Sentiment Analyzer')
st.markdown("Welcome! Enter a movie review below, and the AI will determine if it's positive or negative. Try one of our examples or write your own!")

# --- Example Buttons and Session State ---
if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

col1, col2 = st.columns(2)
if col1.button('Try a Positive Example ✨'):
    st.session_state.review_text = "This movie was absolutely fantastic! The acting was incredible and the story was thrilling from start to finish."
if col2.button('Try a Negative Example 💀'):
    st.session_state.review_text = "It was a complete waste of time. The plot was boring, predictable, and had terrible special effects."

# --- User Input Area ---
user_input = st.text_area(
    'Your Movie Review',
    st.session_state.review_text,
    height=150,
    placeholder="e.g., 'The best movie I have seen all year!'"
)

# --- Main Prediction Logic ---
if st.button('Analyze Sentiment', type="primary"):
    if not user_input.strip():
        st.warning('Please enter a review before analyzing.', icon="⚠️")
    else:
        with st.spinner('The AI is thinking...'):
            preprocessed_input = preprocess_text(user_input, word_index)
            prediction = model.predict(preprocessed_input)
            score = prediction[0][0]
            sentiment = 'Positive' if score > 0.5 else 'Negative'

        st.divider()
        st.subheader("Analysis Result")

        # --- Display Results in Columns ---
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if sentiment == 'Positive':
                st.success("Sentiment: Positive ✅", icon="👍")
            else:
                st.error("Sentiment: Negative ❌", icon="👎")
        
        with col_res2:
            st.metric(label="Prediction Confidence", value=f"{score:.2%}")
            
        st.progress(float(score), text=f"Score: {score:.4f}")
        st.info("This score represents the model's confidence. Scores > 0.5 are classified as Positive.", icon="💡")