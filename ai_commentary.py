import os
import streamlit as st
from google import genai


def get_gemini_api_key() -> str:
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]

    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    raise ValueError("Missing GEMINI_API_KEY. Add it to .streamlit/secrets.toml or environment variables.")


def get_gemini_model() -> str:
    if "GEMINI_MODEL" in st.secrets:
        return st.secrets["GEMINI_MODEL"]
    return os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


@st.cache_data(show_spinner=False, ttl=3600)
def generate_ai_commentary(prompt: str) -> str:
    api_key = get_gemini_api_key()
    model_name = get_gemini_model()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text if hasattr(response, "text") else str(response)