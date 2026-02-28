# import re
# import os
# import numpy as np
# import nltk
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import joblib
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer



# ## NLTK setup

# nltk.download("stopwords")
# nltk.download("wordnet")


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.joblib")

# model = joblib.load(MODEL_PATH)


# lemmatizer = WordNetLemmatizer()
# STOP_WORDS = set(stopwords.words("english"))
# TOKEN_RE = re.compile(r"[A-Za-z]+")

# def clean_text(text: str) -> str:
#     text = "" if text is None else str(text)
#     text = re.sub(r"http\S+|www\S+", " ", text)
#     text = re.sub(r"@\w+", " ", text)
#     text = re.sub(r"#\w+", " ", text)
#     text = re.sub(r"[^A-Za-z\s]", " ", text)
#     words = TOKEN_RE.findall(text.lower())
#     words = [lemmatizer.lemmatize(w) for w in words if w not in STOP_WORDS and len(w) > 1]
#     return " ".join(words)

# def sigmoid(x: float) -> float:
#     return 1.0 / (1.0 + np.exp(-x))

# app = FastAPI(title="Sentiment API", version="1.0")


# # Allow streamlit to call the API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



# class PredictRequest(BaseModel):
#     text: str
    
    
# # make api endpoint for prediction

# @app.get("/health")
# def health_check():
#     return {"status": "Ok"}

# @app.post("/predict")
# def predict(request: PredictRequest):
#     cleaned = clean_text(request.text)
#     pred = int(model.predict([cleaned])[0])
    
    
    
#     #linear: decision_function exist
#     score = float(model.decision_function([cleaned])[0])
#     confidence = float(sigmoid(score))  # pseudo confidence

#     return {
#         "label": pred,
#         "sentiment": "positive" if pred == 1 else "negative",
#         "confidence": confidence,
#         "raw_score": score,
#         "cleaned_text": cleaned,
#     }


import os
import re
import joblib
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬")

@st.cache_resource
def load_stuff():
    nltk.download("stopwords")
    nltk.download("wordnet")

    model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.joblib")
    model = joblib.load(model_path)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    token_re = re.compile(r"[A-Za-z]+")

    return model, lemmatizer, stop_words, token_re

model, lemmatizer, STOP_WORDS, TOKEN_RE = load_stuff()

def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)

    words = TOKEN_RE.findall(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

st.title("💬 Tweet Sentiment Analyzer")
st.write("Type a tweet/text and get Positive or Negative sentiment.")

text = st.text_area("Enter text", height=140, placeholder="e.g. I love this product!")

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text)
        pred = int(model.predict([cleaned])[0])

        score = float(model.decision_function([cleaned])[0])
        conf = sigmoid(score) * 100

        if pred == 1:
            st.success(f"✅ Positive ({conf:.1f}% confidence)")
        else:
            st.error(f"⚠️ Negative ({conf:.1f}% confidence)")

        with st.expander("Details"):
            st.write("Cleaned text:")
            st.code(cleaned)
            st.json({"raw_score": score, "confidence": conf, "label": pred})