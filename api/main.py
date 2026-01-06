import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os
import sys
from pathlib import Path
import pickle

# Add your project root to path so we can import your modules
sys.path.append(str(Path(__file__).parent.parent))

from models.language_model import TransformerLM  # Adjust if your model class name/path is different
from tokenization.tokenizer import Tokenizer  # Your BPE tokenizer

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 600
    temperature: float = 0.85
    top_k: int = 50

app = FastAPI(title="TinyStories GPT Inference API")

# Load model and tokenizer once at startup
@app.on_event("startup")
async def load_model():
    checkpoint_path = "checkpoints/checkpoint_final.pt"
    tokenizer_dir = "data/tokenizer"

    # Load vocab and merges from pickle files
    vocab_path = f"{tokenizer_dir}/vocab.pkl"
    merges_path = f"{tokenizer_dir}/merges.pkl"

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)

    # Now pass the loaded dicts to Tokenizer
    app.state.tokenizer = Tokenizer(vocab, merges)

    print("Tokenizer loaded successfully!")

    # Temporary placeholder for model
    app.state.model = None

async def generate_tokens(request: GenerateRequest):
    tokenizer = app.state.tokenizer
    model = app.state.model

    # Encode prompt
    tokens = tokenizer.encode(request.prompt, return_tensors=True)  # Adjust to your tokenizer API

    # Generation loop (adapt from your generate.py)
    for _ in range(request.max_tokens):
        with torch.no_grad():
            logits = model(tokens)
            # Apply temperature and top-k (copy logic from generate.py)
            # ...
            next_token = ...  # sample next token
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        yielded_text = tokenizer.decode(next_token)
        yield yielded_text

        await asyncio.sleep(0)  # Allow streaming

@app.post("/generate")
async def generate(request: GenerateRequest):
    return StreamingResponse(generate_tokens(request), media_type="text/event-stream")
