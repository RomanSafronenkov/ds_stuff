from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1:22:45 - 2:15:00
# ----------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence_length, embedding dimensionality (n_embd)
        # calculate query, key, value for all heads in a batch and move head forward to be the batch dimension
        # nh is "number of heads", hs is "head_size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=768 channels in the Transformer

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # flash-attention, pytorch will see that it can be compiled
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 1 residual path
        x = x + self.mlp(self.ln_2(x))  # 2 residual path
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # ширина контекста
    vocab_size: int = 50257  # размер словаря: 50000 BPE слияний + 256 токенов байтов + 1 <|endoftext|> токен
    n_layer: int = 12  # число слоев
    n_head: int = 12  # число голов внимания
    n_embd: int = 768  # размер эмбеддинга


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # таблица эмбеддингов токенов
            wpe = nn.Embedding(config.block_size, config.n_embd),  # таблица эмбеддингов позиций токенов
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # последовательные слои трансфорера
            ln_f = nn.LayerNorm(config.n_embd)  # финальная нормализация (нововведение в GPT-2)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # проекция в словарь токенов

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        '''
        init weights like in the original gpt2
        '''
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) **-0.5  # each layer has 2 residual paths, scaling std is needed
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward for the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # override_args = override_args or {} # default to empty dict
        # # only dropout can be overridden see more notes below
        # assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        # if 'dropout' in override_args:
        #     print(f"overriding dropout rate to {override_args['dropout']}")
        #     config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
# ----------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T) -> None:
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        enc = tiktoken.get_encoding('gpt2')
        with open('shakespeare.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# ----------------------------------------------------------------------------------
# attempt to autodetect the device
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=1024)

# ENABLE TF32
torch.set_float32_matmul_precision('high')  # make model faster


# get logits
model = GPT(GPTConfig(vocab_size=50304))  # the dumbiest optimization, make vocab_size a nice number, adding fake tokens, which is divided by 2 many times
model.to(device)
# model = torch.compile(model)  # compile the model to run it a lot faster!, linux only

# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):  # add autocast for bf16 usage, more faster!
        logits, loss = model(x, y)
    # import code; code.interact(local=locals())  # for interraction with code
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()  # wait for gpu finish its work that is scheduled to run
    t1 = time.time()
    dt = (t1 - t0)*1000  # time difference in miliseconds
    tokens_per_second = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_second:.2f}")

import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)




# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

# generate! now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)