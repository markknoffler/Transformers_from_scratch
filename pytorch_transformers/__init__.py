"""
Transformers from Scratch

A complete implementation of GPT from scratch using PyTorch.
"""

__version__ = "0.1.0"
__author__ = "Transformers from Scratch Team"
__email__ = "contact@example.com"

from .models.gpt import GPT, GPTBlock, GPTEmbeddings, GPTLoss
from .configs.gpt_config import GPTConfig

__all__ = [
    "GPT",
    "GPTBlock", 
    "GPTEmbeddings",
    "GPTLoss",
    "GPTConfig",
]
