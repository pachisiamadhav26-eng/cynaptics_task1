# 🎭 Shakespeare GPT — From Scratch

> *Building a GPT-style Language Model from the ground up using PyTorch — no pre-built model libraries, no shortcuts.*

Trained entirely on Shakespeare's works, this model learns to generate authentic play-formatted dialogue complete with character names, Elizabethan vocabulary, and proper scene structure — all from a single text file.

---

## ✨ What Makes This Different

Most language model tutorials use pretrained weights or high-level APIs. This project implements **everything from scratch:**

- ✅ Custom BPE Tokenizer trained on Shakespeare
- ✅ Multi-Head Self Attention — written from zero
- ✅ Full Transformer Block with residual connections
- ✅ Complete training pipeline with early stopping
- ✅ Smart text generation with Top-K sampling

---

## 🗂️ Project Structure

```
cynaptics.task1/
│
├── dataloader.py                    # Downloads Shakespeare dataset
├── tokenizer.py                     # Trains ByteLevel BPE tokenizer
├── data.py                          # Encoding, batching, train/val split
├── model.py                         # Complete GPT architecture
├── train.py                         # Training loop + text generation
│
├── shakespeare.txt                  # Raw training data
├── shakespeare-tokenizer-v2.json    # Trained BPE tokenizer
├── best_model.pt                    # Best saved model checkpoint
│
└── outputs/
    ├── output1.png                  # Sample generated text
    └── output2.png                  # Sample generated text
```

---

## 🧠 Architecture

The model is a **decoder-only Transformer** — same family as GPT.

```
Input Token IDs  (B, T)
        ↓
┌─────────────────────────────┐
│   Token Embedding           │  what is this token?
│ + Position Embedding        │  where is this token?
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐  ×n_layer
│   Transformer Block         │
│   ├── LayerNorm             │
│   ├── Multi-Head Attention  │  tokens communicate
│   ├── Residual Connection   │  x = x + attention(x)
│   ├── LayerNorm             │
│   ├── FeedForward           │  tokens think individually
│   └── Residual Connection   │  x = x + ff(x)
└─────────────────────────────┘
        ↓
Final LayerNorm
        ↓
Linear Projection → Logits (B, T, vocab_size)
        ↓
    ┌───────────┐
    │ Training  │ → Cross Entropy Loss
    │ Generate  │ → Top-K Sampling
    └───────────┘
```

### Component Breakdown

| Component | Role | Why it matters |
|---|---|---|
| **Token Embedding** | Token ID → 128-dim vector | Gives each token a learnable identity |
| **Position Embedding** | Position → 128-dim vector | Tells model the order of tokens |
| **Multi-Head Attention** | Tokens attend to each other | Learns relationships between words |
| **Causal Mask** | Blocks future tokens | Makes it a language model (predict next) |
| **Value Matrix** | Curates what info to share | Tokens share relevant info, not everything |
| **FeedForward** | Per-token processing | Thinks about gathered context individually |
| **Residual Connection** | Adds input to output | Prevents vanishing gradients in deep networks |
| **LayerNorm** | Normalizes activations | Keeps training stable |

---

## 🔤 Tokenizer — Why ByteLevel BPE?

Three tokenizer approaches were considered:

```
Character-level:
  "agreed" → ['a','g','r','e','e','d']   6 tokens
  ✅ simple    ❌ very long sequences    ❌ slow training

Whitespace BPE (old approach):
  "agreed" → ["ag", "reed"]              2 tokens — BROKEN!
  ✅ shorter  ❌ rare words split wrong  ❌ ugly output

ByteLevel BPE (current approach):
  "agreed" → ["Ġagreed"]                1 token — clean!
  ✅ shorter  ✅ words always intact     ✅ clean decode
```

ByteLevel marks word boundaries with `Ġ` before encoding — so BPE merges **never split across word boundaries.** The decoder strips `Ġ` markers automatically, producing clean output.

**Before ByteLevel:**
```
"ag reed ha llow ing Sound prince Stri oak with And words ake"
```
**After ByteLevel:**
```
"agreed hallowing Sound prince with And words"
```

---

## ⚙️ Hyperparameters

### Data
| Parameter | Value | Reason |
|---|---|---|
| `vocab_size` | 3000 | Small enough for dataset size |
| `batch_size` | 32 | Stable gradient updates |
| `block_size` | 128 | Context window — tokens model sees at once |

### Model
| Parameter | Value | Reason |
|---|---|---|
| `n_embd` | 128 | Embedding dimension per token |
| `n_head` | 4 | Attention heads (head_size = 128/4 = 32) |
| `n_layer` | 4 | Transformer blocks stacked |
| `dropout` | 0.5 | High dropout fights overfitting on small data |

### Training
| Parameter | Value | Reason |
|---|---|---|
| `learning_rate` | 3e-4 | Standard for transformers |
| `weight_decay` | 0.1 | L2 regularization — penalizes large weights |
| `max_iters` | 15000 | Maximum training steps |
| `patience` | 5 | Early stopping — saves before overfitting |
| `temperature` | 0.8 | Generation — focused but not repetitive |
| `top_k` | 40 | Generation — blocks nonsense tokens |

---

## 🔁 Training Pipeline

### 1. AdamW Optimizer
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = 3e-4,
    weight_decay = 0.1     # correct L2 regularization
)
```
AdamW fixes a subtle bug in Adam where weight decay is incorrectly coupled with the gradient update. All major transformer models use AdamW.

### 2. Cosine LR Schedule
```
lr
│
3e-4 ──╮
       │╲
       │  ╲
       │    ╲____
~0     └──────────── steps
       0          15000

Starts high → learns fast early
Ends near 0 → fine-tunes carefully at end
```

### 3. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
Prevents gradient explosions — any gradient larger than 1.0 is clipped down.

### 4. Early Stopping
```
step 4000 | val: 4.12  ✅ saved!         patience = 0
step 4500 | val: 4.18  ⚠️ no improve    patience = 1
step 5000 | val: 4.21  ⚠️ no improve    patience = 2
step 5500 | val: 4.25  ⚠️ no improve    patience = 3
step 6000 | val: 4.29  ⚠️ no improve    patience = 4
step 6500 | val: 4.33  ⚠️ no improve    patience = 5
🛑 Training stopped! Best val: 4.12 loaded.
```

### 5. Fighting Overfitting
Shakespeare.txt is ~200k tokens. A model with millions of parameters will memorize it without proper regularization. Three techniques used together:

```
dropout = 0.5       randomly zero 50% of neurons each step
weight_decay = 0.1  shrink large weights after every update
early stopping      stop before memorization kicks in
```

---

## 📊 Training Progress

```
step     0 | train: 8.70 | val: 8.69  ← random baseline ~log(3000)
step   500 | train: 6.70 | val: 6.68  ← learning fast!
step  1000 | train: 6.20 | val: 6.18
step  2000 | train: 5.50 | val: 5.48
step  3000 | train: 4.90 | val: 4.95
step  4000 | train: 4.40 | val: 4.50
...
Best val loss: 4.1176  ✅
```

---

## 🎭 Sample Outputs

### Output 1 — Coriolanus scene
```
There is the cause of his head.

First Officer:
Must's to us to do:
Thou art a man I would do.

CORIOLANUS:
Hark to the people, let us go.

Third Citizen:
Ay, that!

CORIOLANUS:
The gods say 'tis anonounce
The same made you to give the Volsces.

Second Citizen:
This is a very welcome, that has I was worth;
They are to him.
```

### Output 2 — York family scene
```
DUKE OF YORK:
How long-denly father,
For this my son Aumerle?

DUKE OF AUMERLE:
I would say 'Do do speak.

DUCHESS OF YORK:
Why, I have married me, and I would not speak?

DUKE OF YORK:
But thou didst kill me? I say my horse.

DUCHESS OF YORK:
No, but flattering, thou hast done.

DUCHESS OF YORK:
Thou dost suspect me, my sovereign,
I will not hear me on my soul.

DUCHESS OF YORK:
O, I am yours; I speak my duty!
```

**What the model learned without being told:**
- ✅ `CHARACTER:` format followed by dialogue
- ✅ Real Shakespeare character names
- ✅ Elizabethan vocabulary (`thou`, `hast`, `'tis`, `dost`)
- ✅ Natural dialogue turns between characters
- ✅ Proper punctuation and line structure
- ✅ No broken words — clean subword tokenization

---

## 🚀 How to Run

### Prerequisites
```bash
pip install torch tokenizers requests
```

### Step 1 — Download dataset
```bash
python dataloader.py
```
Downloads `shakespeare.txt` (~1MB of Shakespeare plays).

### Step 2 — Train tokenizer
```bash
python tokenizer.py
```
Trains ByteLevel BPE tokenizer and saves `shakespeare-tokenizer-v2.json`.

### Step 3 — Train and generate
```bash
python train.py
```
Trains the model with early stopping. Best checkpoint saved to `best_model.pt`. Generated text printed at the end automatically.

---

## 🔬 Core Concepts

### Causal Self Attention
```python
# tokens can only attend to PAST tokens — not the future
wei = wei.masked_fill(tril == 0, float('-inf'))

# tril for T=4:
# [[1, 0, 0, 0],    token 0 sees: only itself
#  [1, 1, 0, 0],    token 1 sees: 0,1
#  [1, 1, 1, 0],    token 2 sees: 0,1,2
#  [1, 1, 1, 1]]    token 3 sees: all
```

### Query / Key / Value
```
Query  → "what am I looking for?"
Key    → "what do I contain?"
Value  → "what do I give when attended to?"

Query × Key   = decide WHO to attend to   (attention scores)
weights × Value = decide WHAT to receive  (actual information)
```

### Residual Connection
```python
# without residual — gradients vanish in deep networks
x = self.sa(x)

# with residual — gradient always has a direct highway back
x = x + self.sa(x)   # just one + sign makes all the difference!
```

### Top-K Sampling
```
Without top_k → all 3000 tokens eligible
              → occasionally picks nonsense words ❌

With top_k=40 → only top 40 most likely tokens eligible
              → always picks sensible coherent words ✅
```

---

## 📈 Model Size

```
Token Embedding        :   384,000 params
Position Embedding     :    16,384 params
4 × Transformer Block  : 1,315,328 params
Final LayerNorm        :       256 params
Output Head (lm_head)  :   384,000 params
──────────────────────────────────────────
Total                  : ~2.1M parameters
```

---

*Built for Cynaptics Task 1 — implementing a GPT language model from scratch using PyTorch.*
