import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32  # how many independent sequeces will be processed in parallel
block_size = 8  # maximum context length
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # self.token_embedding_table.weight.data *= 0.01  # так можно уменьшить изначальный лосс

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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
            # получим предсказания
            logits, loss = self(idx)
            # интересует только последний шаг по времени
            logits = logits[:, -1, :]  # станет (B, C)
            # применим Softmax, чтобы получить вероятности
            probs = F.softmax(logits, dim=-1)  # также (B, C)
            # сэмплируем из распределения полученных вероятностей
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # добавим новый идекс
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel(vocab_size=vocab_size)
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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))