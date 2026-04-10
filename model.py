import torch
import torch.nn as nn
import torch.nn.functional as F

from data import vocab_size, block_size, n_embd, device, get_batch

#hyperparameters
n_head=4
n_layer=4
dropout=0.3

# single attention head

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

# tril is not a parameter so we write it as buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # attention scores
        wei = q @ k.transpose(-2, -1)          # (B, T, T)
        wei = wei * (C ** -0.5)                # scale down
        wei = wei.masked_fill(
                self.tril[:T, :T] == 0,
                float('-inf')
              )                                # causal mask
        wei = F.softmax(wei, dim=-1)           # (B, T, T)
        wei = self.dropout(wei)

        # weighted sum of values
        v   = self.value(x)                   
        out = wei @ v                       
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
       
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
       
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run all heads in parallel on same input x
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # (B, T, head_size) × num_heads → (B, T, n_embd)
        out = self.dropout(self.proj(out))
        return out
    
#feed forward
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # contract
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size        = n_embd // n_head
        self.sa          = MultiHeadAttention(n_head, head_size)  # self attention
        self.ff          = FeedForward(n_embd)                    # feedforward
        self.ln1         = nn.LayerNorm(n_embd)                   # layer norm 1
        self.ln2         = nn.LayerNorm(n_embd)                   # layer norm 2

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # attention   + residual
        x = x + self.ff(self.ln2(x))   # feedforward + residual
        return x
    

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. embeddings
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 2. transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )

        # 3. final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # 4. output head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. embeddings
        tok_emb = self.token_embedding_table(idx)                    # (B, T, n_embd)
        pos_emb = self.position_embedding_table(                     # (T, n_embd)
                      torch.arange(T, device=device)
                  )
        x = tok_emb + pos_emb                                        # (B, T, n_embd)

        # 2. transformer blocks
        x = self.blocks(x)                                           # (B, T, n_embd)

        # 3. final layer norm
        x = self.ln_f(x)                                             # (B, T, n_embd)

        # 4. output projection → vocabulary scores
        logits = self.lm_head(x)                                     # (B, T, vocab_size)

        # 5. loss calculation
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)       # flatten to (B*T, vocab_size)
            targets = targets.view(B*T)         # flatten to (B*T,)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8 ,top_k=40):
        for _ in range(max_new_tokens):
           
            idx_cond = idx[:, -block_size:]

           
            logits, loss = self(idx_cond)

            
            logits = logits[:, -1, :]            # (B, vocab_size)
            logits=logits/temperature
            values,_ =torch.topk(logits, top_k)
            min_value=values[:, [-1]]
            logits[logits<min_value]=float('-inf')


            
            probs = F.softmax(logits, dim=-1)    # (B, vocab_size)

           
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

           
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)

        return idx
    


