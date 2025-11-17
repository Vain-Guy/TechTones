import streamlit as st
import joblib
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ===
# INITIAL CONFIG
# ===
st.set_page_config(
    page_title="TechTones Sentiment Analyzer",
    page_icon="🎧",
    layout="centered"
)

# Load resources
nltk.download('stopwords')
nltk.download('wordnet')

# ===
# STYLING
# ===
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #f1f1f1;
    }
    .title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #b3b3b3;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .tweet-box textarea {
        border-radius: 10px !important;
        background-color: #1e1e1e !important;
        color: #e5e5e5 !important;
        border: 1px solid #333 !important;
    }
    .footer {
        font-size: 0.8rem;
        color: #666;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===
# HEADER
# ===
st.markdown("<div class='title'>TechTones — Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze public sentiment toward <b>Apple 🍎</b> & <b>Google 🌐</b> in real-time.</div>", unsafe_allow_html=True)

# ===
# LOAD MODEL & VECTORIZER
# ===
@st.cache_resource
def load_assets():
    model = joblib.load("xgboost_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_assets()

# ===
# TEXT PREPROCESSING
# ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# ===
# INPUT
# ===
tweet = st.text_area("💬 Enter a tweet here:", placeholder="E.g. The new Apple iPhone update is incredible!", key="tweet", height=120)

if st.button("🚀 Analyze Sentiment", use_container_width=True):
    if tweet.strip() == "":
        st.warning("Please enter a tweet, Sire.")
    else:
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]

        labels = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😄"}
        sentiment_label = labels.get(prediction, "Unknown")

        st.markdown("---")
        st.subheader("🎯 Prediction")
        st.markdown(f"<h3 style='text-align:center; color:#22c55e;'>{sentiment_label}</h3>", unsafe_allow_html=True)

        st.subheader("📊 Confidence Levels")
        st.progress(float(max(probabilities)))

        st.json({
            "Negative": float(probabilities[0]),
            "Neutral": float(probabilities[1]) if len(probabilities) > 2 else 0.0,
            "Positive": float(probabilities[-1])
        })

        # Dynamic emoji feedback
        if sentiment_label.startswith("Positive"):
            st.balloons()
        elif sentiment_label.startswith("Negative"):
            st.error("😞 This tweet carries negative vibes.")
        else:
            st.info("😐 This one’s pretty neutral.")

# ===
# FOOTER
# ===
st.markdown("<div class='footer'>Built with ❤️ by the TechTones Team — Powered by FastAPI, Streamlit & XGBoost</div>", unsafe_allow_html=True)