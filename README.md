# Mini-GPT: From Scratch

A clean, educational implementation of a GPT-style language model built entirely from scratch using PyTorch. This project covers everything from tokenization and dataset preparation to multi-head attention, transformer blocks, and training — mirroring the architecture of GPT-2.

---

## Architecture

The model follows the decoder-only transformer architecture used in GPT-2:

- **Token + Positional Embeddings** — learned embeddings for both token identity and sequence position
- **Multi-Head Self-Attention** — implemented in `Multi_Head_Attn.py` with causal masking, QKV projections, and dropout
- **Transformer Blocks** — each block stacks LayerNorm → Attention → Residual, then LayerNorm → FeedForward → Residual
- **Language Model Head** — a linear projection from embedding space to vocabulary logits (weight-tied with the token embedding)

---

## Project Structure

```
Mini-GPT--From-Scratch/
├── config.json          # Model hyperparameters
├── model.py             # GPTModel and TransformerBlock definitions
├── Multi_Head_Attn.py   # Multi-head causal self-attention module
├── dataset.py           # Tokenization and DataLoader creation
├── train.py             # Training loop with evaluation
├── utils.py             # LayerNorm, FeedForward, loss helpers, text generation
└── LICENSE              # Unlicense (public domain)
```

---

## Configuration

All hyperparameters are managed via `config.json`:

```json
{
  "device": "cuda",
  "vocab_size": 50257,
  "context_length": 1024,
  "emb_dim": 768,
  "n_heads": 12,
  "n_layers": 12,
  "drop_rate": 0.1,
  "qkv_bias": false
}
```

This matches the GPT-2 (124M) configuration. Adjust `emb_dim`, `n_heads`, and `n_layers` to experiment with smaller or larger variants.

---

## Getting Started

### Prerequisites

```bash
pip install torch tiktoken
```

### Training

Place your training text file (e.g. `the-verdict.txt`) in the project root, then run:

```bash
python train.py
```

The training script will:
1. Load and split the text data (80% train / 20% validation)
2. Create sliding-window dataloaders with a stride of 32 tokens
3. Train for 50 epochs using AdamW with cosine annealing LR
4. Print train/val loss every 5 steps

### Running the Model

To test text generation with an untrained (or pre-loaded) model:

```bash
python model.py
```

This encodes a prompt using the GPT-2 tokenizer (`tiktoken`) and generates 10 new tokens using greedy decoding.

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.0004 |
| Weight Decay | 0.1 |
| LR Schedule | Cosine Annealing |
| Batch Size | 4 |
| Gradient Clipping | max norm 1.0 |
| Epochs | 50 |
| Tokenizer | GPT-2 (tiktoken) |

---

## Key Modules

**`Multi_Head_Attn.py`** — Implements scaled dot-product attention across multiple heads in parallel, with a causal mask to prevent attending to future tokens.

**`utils.py`** — Contains `LayerNorm`, `FeedForward` (with GELU activation), batch loss calculation, model evaluation, and the greedy `generate_text_simple` function.

**`dataset.py`** — Tokenizes raw text and creates sliding-window token batches for language model training.

---

## License

This project is released into the public domain under the [Unlicense](LICENSE). Use it however you like.
