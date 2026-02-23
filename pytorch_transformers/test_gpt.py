import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gpt import GPT

def test_gpt():
    """Test basic GPT functionality"""
    print("Testing GPT implementation...")

    # Create a simple GPT
    model = GPT(
        vocab_size=1000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=512
    )

    # Create dummy input
    batch_size = 4
    seq_len = 32
    src = torch.randint(0, 1000, (batch_size, seq_len))

    # Forward pass
    output = model(src)

    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    print("GPT test passed!")

    return True

if __name__ == "__main__":
    test_gpt()