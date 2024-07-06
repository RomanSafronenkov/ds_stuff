import logging

import tiktoken
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - %(message)s',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)

class DataLoaderShakeSpeare:
    def __init__(self, B, T) -> None:
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        enc = tiktoken.get_encoding('gpt2')
        path = Path(__file__).absolute().parent / 'shakespeare.txt'
        with open(path, 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        _logger.info(f"loaded {len(self.tokens)} tokens")
        _logger.info(f"1 epoch = {len(self.tokens) // (B * T)} batches")

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


class DataLoaderOpenWebText:
    def __init__(self, B, T, split) -> None:
        self.B = B
        self.T = T
        self.split = split
        assert split in {'train', 'val'}

        # at init load tokens from disk and store them in memory
        if split == 'train':
            path = Path(__file__).absolute().parent / 'train.bin'
        elif split == 'val':
            path = Path(__file__).absolute().parent / 'val.bin'

        self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
        _logger.info(f"loaded {len(self.tokens)} tokens")
        _logger.info(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    
    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.from_numpy(buf.astype(np.int64))
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y