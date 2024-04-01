from audioop import bias
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64  # how many independent sequeces will be processed in parallel
block_size = 256  # maximum context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embd = 384
n_head = 6
assert n_embd % n_head == 0
n_layer = 6
dropout = 0.2
# -----------------

torch.manual_seed(1337)

# load data
with open('shakespeare.txt', 'r') as f:
    text = f.read()

# создадим словарь символов
chars = sorted(list(set(text)))
vocab_size = len(chars)

# создадим словари и функции для кодирования и декодирования текстов
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train/test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # генерация маленького батча данных для входов x и таргетов y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # выбираем batch_size случайных мест в данных
    x = torch.stack([data[i:i+block_size] for i in ix])  # с началом в полученных индексах отрезаем + block_size букв
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])# с началом в полученных индексах +1 отрезаем + block_size букв
    return x.to(device), y.to(device)

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
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) = (B, T ,C)
        return out


class MultiHeadAttention(nn.Module):
    """miltiple heads of attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),  # projection
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # + residual connections, layernorm
        x = x + self.ffwd(self.ln2(x))  # + residual connections, layernorm
        return x


# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd//4)
        # self.ffwd = FeedForward(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        # x = self.sa_heads(x)  # apply one head of self-attention, (B, T, C)
        # x = self.ffwd(x)  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx это массив размера (B, T), состоящий из индексов текущего контекста
        for _ in range(max_new_tokens):
            # обрежем вход по длине контекста block_size, дальше него модель не может учитывать
            idx_cond = idx[:, -block_size:]
            # получим предсказания
            logits, loss = self(idx_cond)
            # интересует только последний шаг по времени
            logits = logits[:, -1, :]  # станет (B, C)
            # применим Softmax, чтобы получить вероятности
            probs = F.softmax(logits, dim=-1)  # также (B, C)
            # сэмплируем из распределения полученных вероятностей
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # добавим новый идекс
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

# create optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch
    xb, yb = get_batch('train')

    # evaluate a loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))