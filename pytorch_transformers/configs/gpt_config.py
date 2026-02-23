"""
GPT model configuration
"""

import torch

class GPTConfig:
    """Configuration class for GPT model"""

    def __init__(self):
        # Model architecture
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 6
        self.d_ff = 2048
        self.max_len = 1024
        self.dropout = 0.1

        # Vocabulary settings
        self.vocab_size = None  # Will be set during training

        # Training settings
        self.batch_size = 32
        self.epochs = 10
        self.lr = 1e-4
        self.warmup_steps = 4000
        self.label_smoothing = 0.1

        # Device settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Data settings
        self.max_length = 512

    def update_from_dict(self, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_len': self.max_len,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'warmup_steps': self.warmup_steps,
            'label_smoothing': self.label_smoothing,
            'device': self.device,
            'max_length': self.max_length
        }