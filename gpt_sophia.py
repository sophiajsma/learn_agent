import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3 # Self-attention is more complex than a bigram model, so we should lower the learning rate to avoid divergence.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 6
dropout = 0.2
#------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', enco ding='utf-8') as f:
    text = f.read()   

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join(itos[i] for i in l) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n =  int(0.9 * len(data))# first 90% will be train , rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Since tril is not a parameter, we register it as a buffer.
        # This means it will be part of the state_dict, but not trained by the optimizer.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape # (B, T, n_embd)
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # The reason that we use self.tril[:T, :T] == 0 is to create a mask that has the same shape as wei,
        # but with zeros in the upper triangular part.
        # This way, we can set the upper triangular part of wei to -inf,
        # which will effectively ignore those positions when we apply softmax.
        # It is possible that T < block_size (e.g., at the beginning of a sequence)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        wei = self.dropout(wei)
        
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection layer: heads are concatenated (already n_embd-wide), but each
        # head's output still sits in its own separate chunk. This projection linear 
        # layer lets information from different heads mix/interact before it's written
        # into the residual stream, and gives us a dedicated place to control/scale how much
        # this block contributes to that stream.
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # As mentioned in attention is all you need, the dimensionality of the feedforward layer is typically 4 times the dimensionality of the embedding layer.
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # Similarly to the projection layer in MultiHeadAttention,
            # we can add a linear layer here to control how much this block 
            # contributes to the residual stream.
            nn.Linear(4 * n_embd, n_embd),
            # Add right before add back to the residual stream
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Without residual connections, the model will not be able to learn effectively.
        # x = self.sa(x)
        # x = self.ffwd(x)
        # With residual connections, it can mitigate the vanishing gradient problem and allow for better gradient flow through the network.
        # This is because + means that the gradient can flow directly through the identity path, bypassing the non-linearities and allowing for better learning.
        # x = x + self.sa(x)
        # x = x + self.ffwd(x)
        # The original Transformer's post-norm (LayerNorm(x + Sublayer(x))), which
        # puts LayerNorm directly on the main path and hurts gradient flow in deep stacks.
        # x = self.ln1(x + self.sa(x))
        # x = self.ln2(x + self.ffwd(x))
        # Pre-norm: LayerNorm is applied inside each branch (before self-attention /
        # feedforward), not on the residual path itself. This keeps the residual
        # stream as pure addition so gradients flow straight back unimpeded
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 1. Single head self-attention
        # self.sa_head = Head(n_embd) # head_size = n_embd
        # 2. Multi-head self-attention
        # self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embd//4) # i.e. 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embd)
        # 3. Transformer blocks
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     # Final normalization stabilizes the residual stream before the lm_head.
        #     # In Pre-LN Transformers, the residual stream itself is not normalized after
        #     # the last block, so this norm controls its scale and keeps the resulting
        #     # logits well-conditioned for the vocabulary projection.
        #     nn.LayerNorm(n_embd),
        # ) 
        # 4. Scale
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape 
        # idx and targets are both (B,T) tensor of integers
        token_emb  = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = token_emb + pos_emb # (B, T, n_embd)
        
        # Select one of the following three options for the forward pass:
        # 1. Single head self-attention
        # x = self.sa_head(x) # (B, T, n_embd)
        # 2. Multi-head self-attention
        # x = self.sa_heads(x) # (B, T, n_embd) 
        # x = self.ffwd(x) # (B, T, n_embd)
        # 3. Transformer blocks
        # x = self.blocks(x) # (B, T, n_embd)
        # 4. Scale
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)

        # compute logits and loss
        logits = self.lm_head(x) # (B, T, vocab_size)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond) # logits is (B, T, C)
            # focus only on the last time step
            logits = logits[:, -1, :] # logits is now (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))