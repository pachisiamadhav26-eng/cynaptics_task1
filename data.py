import torch 
from tokenizers import Tokenizer

#encoding in 

batch_size=32
block_size=128
n_embd=128
device='cuda' if torch.cuda.is_available() else 'cpu'

#load tokenizer

tokenizer=Tokenizer.from_file("shakespeare-tokenizer.json")
vocab_size=tokenizer.get_vocab_size()

# --- Load and encode text ---
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

encoded   = tokenizer.encode(text)
token_ids = encoded.ids
data      = torch.tensor(token_ids, dtype=torch.long)

#train/val split
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

#batch generator

def get_batch(split):
    data=train_data if split=='train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i   : i+block_size  ] for i in ix])
    y    = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
            

