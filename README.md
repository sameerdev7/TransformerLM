# TransformerLM - GPT-Style Language Model

A decoder-only Transformer language model trained from scratch on the TinyStories dataset, featuring modern architecture components and comprehensive training infrastructure.

## Overview

This project implements a GPT-style Transformer language model with:
- **Custom BPE Tokenizer** (5,000 vocabulary)
- **Rotary Positional Embeddings (RoPE)** for better position encoding
- **SwiGLU Activation** for improved feed-forward networks
- **RMSNorm** for efficient normalization
- **Multi-Head Self-Attention** with causal masking
- **Complete Training Pipeline** with Weights & Biases integration

## Model Performance

### Training Metrics

| Metric | Value |
|--------|-------|
| **Final Validation Loss** | 1.906 |
| **Final Perplexity** | 6.73 |
| **Training Tokens** | ~77.8M |
| **Total Parameters** | ~18M |
| **Training Iterations** | 5,000 |
| **Best Validation Loss** | 1.906 |

### Training Curves

<div align="center">

| Train Loss | Validation Loss | Validation Perplexity |
|:----------:|:---------------:|:---------------------:|
| ![Train Loss](assets/W&B%20train-loss.png) | ![Val Loss](assets/W&B%20val-loss.png) | ![Val Perplexity](assets/W&B%20val-perplexity.png) |

</div>

**Experiment Tracking**: [View on W&B →](https://wandb.ai/sameer7sayyad-siddhant-college-of-engg/tinystories-gpt/runs/sghwrfsg)

## 🏗️ Architecture

### Model Configuration

```yaml
Model Dimension (d_model):     384
Context Length:                256 tokens
Number of Layers:              6
Attention Heads:               6
Feed-Forward Dimension:        1,536 (4 × d_model)
Vocabulary Size:               5,000
RoPE Theta:                    10,000.0
```

### Architecture Diagram

```
Input Token IDs
     ↓
Token Embedding (vocab_size → d_model)
     ↓
┌─────────────────────────────────────┐
│  Transformer Block (×6 layers)      │
│  ┌───────────────────────────────┐  │
│  │ RMSNorm                        │  │
│  │         ↓                      │  │
│  │ Multi-Head Self-Attention      │  │
│  │ (with RoPE + Causal Mask)     │  │
│  │         ↓                      │  │
│  │ Residual Connection            │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ RMSNorm                        │  │
│  │         ↓                      │  │
│  │ SwiGLU Feed-Forward            │  │
│  │         ↓                      │  │
│  │ Residual Connection            │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
     ↓
Final RMSNorm
     ↓
Output Projection (d_model → vocab_size)
     ↓
Softmax (optional)
     ↓
Output Logits/Probabilities
```

## Project Structure

```
TransformerLM/
├── api/                          # FastAPI deployment
│   └── main.py                   # REST API for inference
├── app.py                        # Streamlit web interface
├── assets/                       # Training visualizations
│   ├── W&B train-loss.png
│   ├── W&B val-loss.png
│   └── W&B val-perplexity.png
├── checkpoints/                  # Model checkpoints
│   └── checkpoint_final.pt       # Final trained model
├── config/                       # Configuration files
│   └── gpt_tinystories.yaml     # Training hyperparameters
├── data/
│   ├── processed/               # Tokenized binary data
│   │   ├── train.bin
│   │   └── val.bin
│   ├── tokenizer/               # BPE tokenizer files
│   │   ├── vocab.pkl
│   │   └── merges.pkl
│   └── preprocessing.py         # Data processing utilities
├── evaluation/
│   └── metrics.py               # Evaluation metrics
├── models/                      # Model architecture
│   ├── language_model.py        # Main TransformerLM class
│   ├── transformer.py           # Transformer block
│   ├── attention.py             # Scaled dot-product attention
│   ├── multihead_attention.py   # Multi-head self-attention
│   ├── embedding.py             # Token embeddings
│   ├── rope.py                  # Rotary positional embeddings
│   ├── swiglu.py               # SwiGLU activation
│   ├── normalization.py         # RMSNorm
│   └── linear.py               # Linear layers
├── scripts/
│   ├── prepare_tinystories.py   # Dataset preparation
│   ├── train_gpt.py            # Training script
│   ├── generate.py             # Text generation
│   └── evaluate_model.py        # Model evaluation
├── tokenization/
│   ├── bpe_trainer.py          # BPE training
│   └── tokenizer.py            # Tokenizer implementation
├── training/
│   ├── optimizer.py            # AdamW optimizer
│   ├── loss.py                 # Cross-entropy loss
│   ├── utils.py                # Training utilities
│   └── wandb_logger.py         # W&B integration
└── requirements.txt
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TransformerLM.git
cd TransformerLM

# Install dependencies
pip install -r requirements.txt
```

### Generate Text

```bash
python scripts/generate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --tokenizer-dir data/tokenizer \
    --prompt "Once upon a time, there was a cheerful little rabbit named Hopper who" \
    --max-tokens 600 \
    --temperature 0.85 \
    --top-k 50
```

### Run Web Interface (Streamlit)

```bash
# Start the FastAPI backend
uvicorn api.main:app --reload --port 8000

# In a new terminal, start the Streamlit frontend
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Run API Server Only

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

API Endpoints:
- `POST /generate` - Generate text from a prompt
- `GET /health` - Health check

## Training from Scratch

### 1. Prepare the Dataset

```bash
python scripts/prepare_tinystories.py \
    --vocab-size 5000 \
    --max-samples 500000
```

This will:
- Download TinyStories from HuggingFace
- Train a BPE tokenizer with 5,000 vocabulary
- Tokenize and save the dataset as binary files

### 2. Train the Model

```bash
python scripts/train_gpt.py \
    --config config/gpt_tinystories.yaml
```

Training configuration (`config/gpt_tinystories.yaml`):
```yaml
training:
  batch_size: 64
  learning_rate: 3.0e-4
  min_lr: 1.0e-4
  max_iters: 5000
  warmup_iters: 500
  grad_clip: 1.0
```

### 3. Evaluate the Model

```bash
python scripts/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --save-report
```

## Evaluation Metrics

The model evaluation includes:

### Automatic Metrics
- **Perplexity**: 6.73 (lower is better)
- **Cross-Entropy Loss**: 1.906
- **Token Accuracy**: Measures next-token prediction accuracy

### Generation Quality Metrics
- **Diversity Metrics**: 
  - Unique unigram ratio
  - Unique bigram ratio
  - Unique trigram ratio
  - Vocabulary size
  - Average text length

- **Repetition Metrics**: 
  - 2-gram repetition rate
  - 3-gram repetition rate
  - 4-gram repetition rate

- **Coherence Score**: 
  - Sentence structure analysis
  - Capitalization patterns
  - Length variance

- **BLEU Score**: (when reference texts available)

### Sample Evaluation Output

```
======================================================================
EVALUATION SUMMARY
======================================================================
✓ Perplexity: 6.73
✓ Token Accuracy: 42.15%
✓ Avg Text Length: 45.3 tokens
✓ Unique Unigrams: 68.4%
✓ Coherence Score: 78.2%
```

## Sample Outputs

**Prompt**: *"Once upon a time, there was a cheerful little rabbit named Hopper who"*

**Generated**:
```
Once upon a time, there was a cheerful little rabbit named Hopper who 
lived in a cozy burrow. One sunny day, Hopper decided to explore the 
big forest. As he hopped along, he met a friendly squirrel named Sam. 
They played together and had lots of fun...
```

**Prompt**: *"One day, a little girl"*

**Generated**:
```
One day, a little girl named Lily went to the park with her mom. She 
saw a big red ball and wanted to play with it. Lily kicked the ball 
very high and it went over the fence...
```

## 🔧 Model Components

### Custom Implementations

All core components are implemented from scratch:

1. **Scaled Dot-Product Attention** (`models/attention.py`)
   - Manual computation of attention scores
   - Causal masking for autoregressive generation
   - Numerically stable softmax

2. **RoPE (Rotary Positional Embedding)** (`models/rope.py`)
   - Rotates query and key vectors based on position
   - Better length extrapolation than learned embeddings
   - Precomputed frequency caching

3. **SwiGLU Activation** (`models/swiglu.py`)
   - Gated linear unit with SiLU activation
   - Improved over standard ReLU/GELU
   - Optimal dimension scaling (8/3 × d_model)

4. **RMSNorm** (`models/normalization.py`)
   - Simpler and faster than LayerNorm
   - No bias or mean centering
   - Learnable gain parameter

5. **Custom BPE Tokenizer** (`tokenization/`)
   - Byte-pair encoding from scratch
   - GPT-2 style regex splitting pattern
   - Special token support

6. **AdamW Optimizer** (`training/optimizer.py`)
   - Decoupled weight decay
   - Bias correction
   - Custom implementation

## 📊 Training Configuration

Key hyperparameters used:

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate (max) | 3e-4 |
| Learning Rate (min) | 1e-4 |
| Weight Decay | 0.1 |
| Gradient Clipping | 1.0 |
| Warmup Steps | 500 |
| Total Steps | 5,000 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| LR Schedule | Cosine with warmup |

## 🛠️ Requirements

```
torch>=2.0.0
numpy>=1.24.0
regex>=2023.0.0
datasets>=2.14.0
pyyaml>=6.0
tqdm>=4.65.0
psutil>=5.9.0
matplotlib>=3.7.0
wandb>=0.15.0
scipy>=1.10.0
scikit-learn>=1.3.0
fastapi>=0.100.0
streamlit>=1.25.0
uvicorn>=0.23.0
```

## 📖 Usage Examples

### Python API

```python
import torch
from models.language_model import TransformerLM
from tokenization.tokenizer import Tokenizer
import pickle

# Load tokenizer
with open("data/tokenizer/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("data/tokenizer/merges.pkl", "rb") as f:
    merges = pickle.load(f)
tokenizer = Tokenizer(vocab, merges)

# Load model
checkpoint = torch.load("checkpoints/checkpoint_final.pt")
model = TransformerLM(
    vocab_size=5000,
    context_length=256,
    d_model=384,
    num_layers=6,
    num_heads=6,
    d_ff=1536,
    rope_theta=10000.0
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt)
input_tensor = torch.tensor([input_ids])

with torch.no_grad():
    for _ in range(100):  # Generate 100 tokens
        logits = model(input_tensor, apply_softmax=False)
        next_token = torch.argmax(logits[0, -1, :])
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

generated_text = tokenizer.decode(input_tensor[0].tolist())
print(generated_text)
```

### REST API

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.85,
        "top_k": 50
    },
    stream=True
)

for chunk in response.iter_content(decode_unicode=True):
    print(chunk, end='', flush=True)
```

## Key Features

### 1. Modern Architecture
- Pre-normalization transformer blocks
- Rotary positional embeddings for better position awareness
- SwiGLU activation for improved expressiveness
- RMSNorm for faster and simpler normalization

### 2. Efficient Training
- Gradient clipping for stability
- Cosine learning rate schedule with warmup
- AdamW optimizer with weight decay
- Mixed precision support (when available)

### 3. Comprehensive Evaluation
- Perplexity and loss metrics
- Token-level accuracy
- Generation quality analysis
- Diversity and repetition detection
- Coherence scoring

### 4. Production Ready
- FastAPI REST API
- Streamlit web interface
- Streaming text generation
- Model checkpointing
- W&B experiment tracking

---

<div align="center">
Made with ❤️ using PyTorch
</div>
