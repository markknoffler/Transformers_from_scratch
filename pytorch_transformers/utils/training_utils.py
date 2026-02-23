import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import time
import math

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate token-level accuracy"""
    with torch.no_grad():
        # Get predicted tokens
        _, predicted = torch.max(predictions, dim=-1)

        # Create mask for non-padding tokens
        mask = (targets != 0)  # Assuming 0 is PAD token

        # Calculate accuracy
        correct = (predicted == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()

        return accuracy.item() if not math.isnan(accuracy.item()) else 0.0

def train_epoch(model: nn.Module, dataloader, optimizer, loss_fn, device: str,
               epoch: int, print_every: int = 100) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(src)  # GPT only needs source input

        # Calculate loss - we need to compare output with target shifted by one
        loss = loss_fn(output, tgt)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Calculate accuracy
        accuracy = calculate_accuracy(output, tgt)

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        if batch_idx % print_every == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch}, Batch {batch_idx}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {accuracy:.4f}, '
                  f'Time: {elapsed:.2f}s')

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }

def validate_epoch(model: nn.Module, dataloader, loss_fn, device: str) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            output = model(src)  # GPT only needs source input

            # Calculate loss
            loss = loss_fn(output, tgt)

            # Calculate accuracy
            accuracy = calculate_accuracy(output, tgt)

            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }

def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float,
                   filepath: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model: nn.Module, optimizer, filepath: str):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from {filepath}")
    return epoch, loss