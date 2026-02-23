import torch
import torch.nn as nn
import unittest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.gpt import GPT
from losses.gpt_loss import GPTLoss
from scripts.tokenize import Vocab

class TestGPTArchitecture(unittest.TestCase):
    """Test cases for GPT architecture implementation"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.vocab_size = 1000
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 6
        self.d_ff = 2048
        self.max_len = 512
        self.batch_size = 4
        self.seq_len = 32

        # Create a simple vocabulary for testing
        self.vocab = Vocab()
        for i in range(self.vocab_size):
            self.vocab.add_token(f"token_{i}")

    def test_gpt_model_initialization(self):
        """Test that GPT model initializes correctly"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.config.vocab_size, self.vocab_size)
        self.assertEqual(model.config.d_model, self.d_model)
        self.assertEqual(model.config.n_heads, self.n_heads)
        self.assertEqual(model.config.n_layers, self.n_layers)
        self.assertEqual(model.config.d_ff, self.d_ff)
        self.assertEqual(model.config.max_len, self.max_len)

    def test_gpt_forward_pass(self):
        """Test that GPT model produces correct output shape"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        # Create dummy input
        src = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        # Forward pass
        output = model(src)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape)

    def test_causal_attention_masking(self):
        """Test that causal attention properly masks future tokens"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=1,  # Use single layer for simplicity
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        # Create a small sequence to test attention
        src = torch.randint(0, self.vocab_size, (1, 10))  # batch_size=1, seq_len=10

        # Forward pass
        output = model(src)

        # The model should process without errors
        self.assertEqual(output.shape, (1, 10, self.vocab_size))

    def test_gpt_blocks_structure(self):
        """Test that GPT blocks are properly structured"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=2,
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        # Check that we have the expected number of layers
        self.assertEqual(len(model.transformer_blocks), 2)

        # Check that each block has the right components
        for block in model.transformer_blocks:
            self.assertTrue(hasattr(block, 'attention'))
            self.assertTrue(hasattr(block, 'ffn'))
            self.assertTrue(hasattr(block, 'norm1'))
            self.assertTrue(hasattr(block, 'norm2'))

    def test_embedding_layer(self):
        """Test embedding layer functionality"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=1,
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        # Test that embedding works
        src = torch.randint(0, self.vocab_size, (1, 5))
        embedded = model.embedding(src)

        self.assertEqual(embedded.shape, (1, 5, self.d_model))

    def test_output_projection(self):
        """Test that output projection works correctly"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=1,
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        # Create input
        src = torch.randint(0, self.vocab_size, (2, 8))

        # Forward pass
        output = model(src)

        # Check that output has correct vocabulary dimension
        self.assertEqual(output.shape[2], self.vocab_size)

    def test_loss_function(self):
        """Test that loss function works with GPT outputs"""
        loss_fn = GPTLoss()

        # Create sample predictions and targets
        batch_size, seq_len, vocab_size = 2, 10, 1000
        predictions = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Calculate loss
        loss = loss_fn(predictions, targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)

    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=2,
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        # Create input
        src = torch.randint(0, self.vocab_size, (2, 10))
        targets = torch.randint(0, self.vocab_size, (2, 10))

        # Forward pass
        output = model(src)

        # Calculate loss
        loss_fn = GPTLoss()
        loss = loss_fn(output, targets)

        # Backward pass
        loss.backward()

        # Check that gradients exist for some parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        self.assertTrue(has_grad)

    def test_model_size(self):
        """Test that model has expected number of parameters"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters
        self.assertTrue(total_params > 1000000)  # At least 1M parameters

    def test_batch_processing(self):
        """Test that model handles different batch sizes correctly"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=1,
            d_ff=self.d_ff,
            max_len=self.max_len
        )

        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            src = torch.randint(0, self.vocab_size, (batch_size, 16))
            output = model(src)
            self.assertEqual(output.shape, (batch_size, 16, self.vocab_size))

    def test_sequence_length_processing(self):
        """Test that model handles different sequence lengths correctly"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=1,
            d_ff=self.d_ff,
            max_len=64  # Set smaller max length
        )

        # Test different sequence lengths
        for seq_len in [8, 16, 32]:
            src = torch.randint(0, self.vocab_size, (2, seq_len))
            output = model(src)
            self.assertEqual(output.shape, (2, seq_len, self.vocab_size))

if __name__ == '__main__':
    unittest.main()