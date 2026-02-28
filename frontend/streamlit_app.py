import requests
import streamlit as st
import os


DEFAULT_API = "http://127.0.0.1:8000"

try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", DEFAULT_API)
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="centered")

# # Change this when deployed:
# API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")

st.title("💬 Tweet Sentiment Analyzer")
st.write("Enter a tweet/text and get **Positive / Negative** sentiment.")

text = st.text_area("Enter text", height=150, placeholder="Type something like: I love this product!")

col1, col2 = st.columns([1, 1])

with col1:
    analyze = st.button("Analyze", type="primary")

with col2:
    st.caption(f"API: `{API_URL}`")

if analyze:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=20)
                resp.raise_for_status()
                data = resp.json()

                sentiment = data["sentiment"]
                conf = data["confidence"] * 100

                if sentiment == "positive":
                    st.success(f"✅ Positive ({conf:.1f}% confidence)")
                else:
                    st.error(f"⚠️ Negative ({conf:.1f}% confidence)")

                with st.expander("See details"):
                    st.json(data)

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach API. Is FastAPI running on the correct URL?")
            except requests.exceptions.Timeout:
                st.error("⏱️ API timeout. Try again.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")