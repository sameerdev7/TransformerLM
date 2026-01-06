import streamlit as st
import requests

st.title("TinyStories GPT Story Generator")

# Sidebar for params
with st.sidebar:
    st.header("Generation Settings")
    max_tokens = st.slider("Max tokens", 100, 1000, 600)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.85)
    top_k = st.slider("Top-k", 1, 200, 50)

prompt = st.text_input("Prompt", "Once upon a time, there was a cheerful little rabbit named Hopper who")

if st.button("Generate Story"):
    if prompt:
        with st.spinner("Generating..."):
            response = requests.post(
                "http://localhost:8000/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                },
                stream=True,
            )

            if response.status_code == 200:
                placeholder = st.empty()
                generated = ""
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    generated += chunk
                    placeholder.markdown(generated + "▌")  # Cursor effect
                placeholder.markdown(generated)
            else:
                st.error("Error from backend")
