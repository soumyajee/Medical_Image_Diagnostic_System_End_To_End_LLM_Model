import streamlit as st
import requests
import io
from PIL import Image
import base64
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import pandas as pd
import torch

# Load environment variables
load_dotenv()

# Load Embedding Model Properly
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

st.title("Medical Imaging Diagnosis Agent")

uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    api_key = st.text_input("Enter OpenAI API Key", type="password")

    if st.button("Analyze Image"):
        if api_key:
            with st.spinner("Analyzing..."):
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert AI radiologist analyzing medical images..."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this image in detail with abnormalities, observations & recommendations."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                                }
                            ]
                        }
                    ]
                }

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(result['choices'][0]['message']['content'])
                else:
                    st.error("Failed to get response from OpenAI API")

        else:
            st.warning("Please enter your OpenAI API key.")

