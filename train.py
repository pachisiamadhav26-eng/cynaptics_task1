import torch
from model import GPTLanguageModel
from data import get_batch, device
from tokenizers import Tokenizer

# Training Hyperparameters 
max_iters     = 15000   
eval_interval = 500     
eval_iters    = 200     
learning_rate = 3e-4   
weight_decay  = 0.1     
patience      = 5       


#  Initialize model
model = GPTLanguageModel().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Optimizer

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = learning_rate,
    weight_decay = weight_decay
)

#  Cosine LR Scheduler

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max = max_iters
)

#  Loss Estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()                        
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y         = get_batch(split)
            logits, loss = model(X, Y)
            losses[k]    = loss.item()
        out[split] = losses.mean()
    model.train()                       
    return out

# Training Loop
best_val_loss  = float('inf')
patience_count = 0

for step in range(max_iters):

    # evaluate every eval_interval steps
    if step % eval_interval == 0:
        losses = estimate_loss()
        gap    = losses['val'] - losses['train']
        print(f"step {step:5d} | train: {losses['train']:.4f} | val: {losses['val']:.4f} | gap: {gap:.4f}", end="")

        #  early stopping
        if losses['val'] < best_val_loss:
            best_val_loss  = losses['val']
            patience_count = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("saved!")
        else:
            patience_count += 1
            print(f" no improve ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"\n early stopping at step {step}!")
                print(f"  best val loss: {best_val_loss:.4f}")
                break
      

    # get batch
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)

    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()    


print("\nLoading best model...")
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
print(f"Best val loss: {best_val_loss:.4f}")

#  Generate Text
tokenizer = Tokenizer.from_file("shakespeare-tokenizer-v2.json")

context       = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(
    context,
    max_new_tokens = 300,
    temperature    = 0.8,   
    top_k          = 40     
)[0].tolist()

print("\n── Generated Text ──────────────────")
print(tokenizer.decode(generated_ids))
