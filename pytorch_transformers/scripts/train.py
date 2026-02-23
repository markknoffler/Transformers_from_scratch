import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from typing import Dict, Any
import time

# Add the project root to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.gpt import GPT
from losses.gpt_loss import GPTLoss
from data.data_utils import prepare_data
from scripts.tokenize import Vocab
from scripts.evaluate import evaluate_model

def train_model(model: nn.Module, dataloader: DataLoader,
                optimizer: optim.Optimizer, loss_fn: nn.Module,
                device: str, epochs: int = 10) -> Dict[str, Any]:
    """Train the GPT model

    Args:
        model: GPT model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        loss_fn: Loss function
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs

    Returns:
        Training metrics dictionary
    """
    model.train()
    model.to(device)

    train_metrics = {
        'losses': [],
        'accuracies': [],
        'perplexities': []
    }

    total_steps = len(dataloader) * epochs

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_perplexity = 0.0
        num_batches = 0

        print(f"Starting epoch {epoch + 1}/{epochs}")

        for step, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(src)

            # Calculate loss
            loss = loss_fn(output, tgt)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Calculate metrics
            accuracy = calculate_accuracy(output, tgt)
            perplexity = torch.exp(loss).item()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            epoch_perplexity += perplexity
            num_batches += 1

            # Print progress
            if (step + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                avg_accuracy = epoch_accuracy / num_batches
                print(f"Epoch {epoch + 1}, Step {step + 1}, "
                      f"Loss: {avg_loss:.4f}, "
                      f"Accuracy: {avg_accuracy:.4f}, "
                      f"Perplexity: {avg_perplexity:.4f}")

        # Calculate average metrics for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches
        avg_epoch_perplexity = epoch_perplexity / num_batches

        train_metrics['losses'].append(avg_epoch_loss)
        train_metrics['accuracies'].append(avg_epoch_accuracy)
        train_metrics['perplexities'].append(avg_epoch_perplexity)

        print(f"Epoch {epoch + 1} completed:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Average Accuracy: {avg_epoch_accuracy:.4f}")
        print(f"  Average Perplexity: {avg_epoch_perplexity:.4f}")
        print("-" * 50)

    return train_metrics

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate token-level accuracy"""
    with torch.no_grad():
        # Get predicted tokens
        _, predicted = torch.max(predictions, dim=-1)

        # Create mask for non-padding tokens (assuming 0 is PAD token)
        mask = (targets != 0)

        # Calculate accuracy
        correct = (predicted == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()

        return accuracy.item() if not torch.isnan(accuracy) else 0.0

def save_model(model: nn.Module, vocab: Vocab, save_path: str):
    """Save model and vocabulary

    Args:
        model: GPT model to save
        vocab: Vocabulary object
        save_path: Path to save model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model state
    model_path = os.path.join(save_path, 'model.pth')
    torch.save(model.state_dict(), model_path)

    # Save vocabulary
    vocab_path = os.path.join(save_path, 'vocab.json')
    vocab_dict = {
        'word2idx': vocab.word2idx,
        'idx2word': vocab.idx2word,
        'idx': vocab.idx
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_dict, f)

def load_model(model_path: str, vocab: Vocab, device: str = 'cpu') -> nn.Module:
    """Load a saved model

    Args:
        model_path: Path to model file
        vocab: Vocabulary object
        device: Device to load model on

    Returns:
        Loaded GPT model
    """
    # Create model instance
    model = GPT(
        vocab_size=len(vocab),
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=512
    )

    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GPT model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--save_path', type=str, default='./models',
                       help='Path to save model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Vocabulary size')

    args = parser.parse_args()

    print("Starting GPT model training...")
    print(f"Device: {args.device}")
    print(f"Data path: {args.data_path}")
    print(f"Save path: {args.save_path}")

    # Create vocabulary
    vocab = Vocab()

    # Load data
    print("Loading data...")
    dataloader = prepare_data(args.data_path, vocab, args.batch_size)

    # Create model
    print("Creating model...")
    model = GPT(
        vocab_size=len(vocab),
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=512
    )

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Create loss function
    loss_fn = GPTLoss(label_smoothing=0.1)

    # Train model
    print("Starting training...")
    start_time = time.time()

    train_metrics = train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        epochs=args.epochs
    )

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final training metrics:")
    print(f"  Loss: {train_metrics['losses'][-1]:.4f}")
    print(f"  Accuracy: {train_metrics['accuracies'][-1]:.4f}")
    print(f"  Perplexity: {train_metrics['perplexities'][-1]:.4f}")

    # Save model
    print("Saving model...")
    save_model(model, vocab, args.save_path)
    print("Model saved successfully!")

if __name__ == '__main__':
    main()