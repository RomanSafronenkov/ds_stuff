import os
import math
import time
import logging

import torch
import torch.nn.functional as F
import tiktoken

from model import GPTConfig, GPT
from data.dataloader import DataLoaderOpenWebText

# _MODE = 'from_scratch'  # select mode if needed 'resume'
_MODE = 'resume'


logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - %(message)s',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)


# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

fileHandler = logging.FileHandler(log_file)
_logger.addHandler(fileHandler)

# ----------------------------------------------------------------------------------
# attempt to autodetect the device

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
_logger.info(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

_logger.info(f"current mode: {_MODE}")

total_batch_size = 524288  # 2**19 ~ 0.5M in number of tokens
B = 4
T = 1024
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
_logger.info(f'total desired batch_size: {total_batch_size}')
_logger.info(f'=> calculated gradient accumulation steps: {grad_accum_steps}')

train_loader = DataLoaderOpenWebText(B=B, T=T, split='train')
val_loader = DataLoaderOpenWebText(B=B, T=T, split='val')

# ENABLE TF32
torch.set_float32_matmul_precision('high')  # make model faster

# get logits
model = GPT(GPTConfig(vocab_size=50304))  # the dumbiest optimization, make vocab_size a nice number, adding fake tokens, which is divided by 2 many times
model.to(device)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 * 3  # 10B tokens / 524288 batch size is ~ 19073 iterations for one epoch, actually 17234
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def save_checkpoint(model, optimizer, step, val_loss_accum):
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),  # to continue training
        'config': model.config,
        'step': step,
        'val_loss': val_loss_accum.item()
    }
    torch.save(checkpoint, checkpoint_path)

def evaluate_val(model, val_dataloader):
    model.eval()
    val_dataloader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    return val_loss_accum


# optimize!
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

enc = tiktoken.get_encoding('gpt2')
init_step = 0

if _MODE == 'resume':
    # set the checkpoint to load
    load_ckpt_path = os.path.join(log_dir, "model_17500.pt")
    _logger.info(f"loading checkpoint: {load_ckpt_path}")

    checkpoint = torch.load(load_ckpt_path, map_location=device)

    # 17233 is max current_position for openwebtext dataloader, one epoch size
    if checkpoint['step'] >= 17234:
        resume_cur_pos = checkpoint['step'] - 17234 * (checkpoint['step'] // 17234)
    else:
        resume_cur_pos = checkpoint['step']

    train_loader.current_position = resume_cur_pos * total_batch_size  # return to the position of the training
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    init_step = checkpoint['step']

# ----------------------------------------------------------------------------------
try:
    for step in range(init_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            val_loss_accum = evaluate_val(model, val_loader)
            _logger.info(f"validation loss: {val_loss_accum.item():.4f}")

            if step > 0 and (step % 2500 == 0 or last_step):
                # write checkpoints
                save_checkpoint(model, optimizer, step, val_loss_accum)

        # once in a while we generate samples from the model
        if (step > 0 and step % 250 == 0) or last_step:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)
            while xgen.size(1) < max_length:
                # forward model to get the logits
                with torch.no_grad():
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from top-k probabilities
                    # note: miltinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                _logger.info(f"sample {i}: {decoded}")

        # training loop
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):  # grad accumulation
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):  # add autocast for bf16 usage, more faster!
                logits, loss = model(x, y)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps  # fix the loss issue with reduction="mean"
            loss_accum += loss.detach()
            loss.backward()  # += grad
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip like gpt3
        optimizer.step()
 
        # lr scheduling
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        torch.cuda.synchronize()  # wait for gpu finish its work that is scheduled to run
        t1 = time.time()
        dt = (t1 - t0)  # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_second = tokens_processed / dt
        _logger.info(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} norm: {norm:.4f} | dt: {dt:.2f}s | tok/sec: {tokens_per_second:.2f}")
except KeyboardInterrupt:
    _logger.info("TRAINING INTERRUPTED BY USER, SAVING MODEL")

    val_loss_accum = evaluate_val(model, val_loader)
    _logger.info(f"validation loss: {val_loss_accum.item():.4f}")

    # save checkpoints
    save_checkpoint(model, optimizer, step, val_loss_accum)