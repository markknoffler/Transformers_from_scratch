import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import json
import os

class Vocab:
    """Simple vocabulary class for tokenization"""

    def __init__(self, specials: List[str] = None):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Add special tokens
        if specials is None:
            specials = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

        for token in specials:
            self.add_token(token)

    def add_token(self, token: str):
        """Add a token to the vocabulary"""
        if token not in self.word2idx:
            self.word2idx[token] = self.idx
            self.idx2word[self.idx] = token
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def __call__(self, token: str):
        """Get index of token"""
        return self.word2idx.get(token, self.word2idx.get('<UNK>'))

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices back to tokens"""
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]

class TextDataset:
    """Dataset for text data processing"""

    def __init__(self, texts: List[str], vocab: Vocab, max_length: int = 512):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize text
        tokens = text.split()
        indices = [self.vocab(token) for token in tokens]

        # Add BOS and EOS tokens
        indices = [self.vocab('<BOS>')] + indices + [self.vocab('<EOS>')]

        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.vocab('<PAD>')] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]

        # Convert to tensors
        src = torch.tensor(indices[:-1], dtype=torch.long)  # All but last token
        tgt = torch.tensor(indices[1:], dtype=torch.long)   # All but first token

        return src, tgt

def load_data(data_path: str) -> List[str]:
    """Load text data from file"""
    texts = []

    if os.path.isdir(data_path):
        # Load from directory of text files
        for filename in os.listdir(data_path):
            if filename.endswith('.txt'):
                with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                    texts.extend(f.readlines())
    else:
        # Load from single file
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()

    return texts

def tokenize_text(text: str, vocab: Vocab) -> torch.Tensor:
    """Tokenize a single text string"""
    tokens = text.split()
    indices = [vocab(token) for token in tokens]
    return torch.tensor(indices, dtype=torch.long)

def detokenize_text(indices: List[int], vocab: Vocab) -> str:
    """Convert token indices back to text"""
    tokens = vocab.decode(indices)
    return ' '.join(tokens)