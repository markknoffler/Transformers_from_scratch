import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

class GPT(nn.Module):
    """Complete GPT implementation from scratch"""

    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout

        # Create configuration
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'max_len': max_len,
            'dropout': dropout
        })()

        # Embeddings
        self.embedding = GPTEmbeddings(vocab_size, d_model, max_len, dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GPT model"""
        # Get embeddings
        x = self.embedding(src)

        # Create causal mask
        seq_len = src.size(1)
        mask = self._create_causal_mask(seq_len, device=src.device)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Output projection
        output = self.output_projection(x)

        return output

    def _create_causal_mask(self, seq_len: int, device: str = 'cpu') -> torch.Tensor:
        """Create causal mask for attention"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.uint8))
        return mask.unsqueeze(0).unsqueeze(1)  # Add batch and head dimensions

class GPTEmbeddings(nn.Module):
    """GPT embeddings with token and positional embeddings"""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize positional embeddings
        self._init_positional_embeddings(max_len, d_model)

    def _init_positional_embeddings(self, max_len: int, d_model: int):
        """Initialize positional embeddings using sinusoidal encoding"""
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass through embeddings"""
        seq_len = src.size(1)

        # Get token embeddings
        token_emb = self.token_embedding(src)

        # Get positional embeddings
        pos_emb = self.pe[:seq_len, :].unsqueeze(0)

        # Combine embeddings
        x = token_emb + pos_emb

        # Apply dropout
        x = self.dropout(x)

        return x

class GPTBlock(nn.Module):
    """Single GPT block with attention and feed-forward"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # Multi-head attention with causal masking
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through GPT block"""
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x

class GPTLoss(nn.Module):
    """Custom loss function for GPT training"""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass through loss function"""
        # Standard cross entropy loss
        batch_size, seq_len, vocab_size = predictions.shape
        predictions = predictions.view(-1, vocab_size)
        targets = targets.view(-1)

        # Calculate loss without reduction
        loss = self.criterion(predictions, targets)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Calculate the smoothed loss
            smooth_loss = -torch.sum(F.log_softmax(predictions, dim=1), dim=1)
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        # Return mean loss
        return loss.mean()