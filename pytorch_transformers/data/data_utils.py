import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
import json

class TextDataset(Dataset):
    """Dataset for text data processing"""

    def __init__(self, texts: List[str], vocab, max_length: int = 512,
                 bos_token: str = '<BOS>', eos_token: str = '<EOS>'):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize text
        tokens = text.split()
        indices = [self.vocab(token) for token in tokens]

        # Add BOS and EOS tokens
        indices = [self.vocab(self.bos_token)] + indices + [self.vocab(self.eos_token)]

        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([0] * (self.max_length - len(indices)))  # 0 is PAD token
        else:
            indices = indices[:self.max_length]

        # Convert to tensors
        src = torch.tensor(indices[:-1], dtype=torch.long)  # All but last token
        tgt = torch.tensor(indices[1:], dtype=torch.long)   # All but first token

        return src, tgt

def load_text_data(data_path: str) -> List[str]:
    """Load text data from file or directory

    Args:
        data_path: Path to text file or directory containing text files

    Returns:
        List of text strings
    """
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

def create_dataloader(texts: List[str], vocab, batch_size: int = 32,
                     max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for text data

    Args:
        texts: List of text strings
        vocab: Vocabulary object
        batch_size: Batch size for training
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data

    Returns:
        DataLoader for text data
    """
    dataset = TextDataset(texts, vocab, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def prepare_data(data_path: str, vocab, batch_size: int = 32,
                max_length: int = 512) -> DataLoader:
    """Prepare data for training

    Args:
        data_path: Path to data
        vocab: Vocabulary object
        batch_size: Batch size for training
        max_length: Maximum sequence length

    Returns:
        Prepared DataLoader
    """
    # Load texts
    texts = load_text_data(data_path)

    # Create dataloader
    dataloader = create_dataloader(texts, vocab, batch_size, max_length)

    return dataloader