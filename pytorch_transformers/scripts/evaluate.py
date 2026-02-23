import torch
import torch.nn as nn
import math
from typing import List, Tuple
import numpy as np

def calculate_perplexity(model: nn.Module, dataloader, device: str) -> float:
    """Calculate perplexity of the model on a dataset"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            output = model(src)

            # Calculate loss
            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            batch_loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))

            # Count tokens (excluding padding)
            num_tokens = (tgt != 0).sum().item()

            total_loss += batch_loss.item()
            total_tokens += num_tokens

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity

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

def evaluate_model(model: nn.Module, dataloader, device: str) -> dict:
    """Comprehensive model evaluation"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_perplexity = 0.0
    num_batches = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            output = model(src)

            # Calculate loss
            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))

            # Calculate accuracy
            accuracy = calculate_accuracy(output, tgt)

            # Calculate perplexity
            num_tokens = (tgt != 0).sum().item()
            avg_loss = loss.item() / num_tokens
            perplexity = math.exp(avg_loss)

            total_loss += loss.item()
            total_accuracy += accuracy
            total_perplexity += perplexity
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_perplexity = total_perplexity / num_batches

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'perplexity': avg_perplexity
    }

def generate_text(model: nn.Module, vocab, start_text: str, max_length: int = 50,
                  temperature: float = 1.0, device: str = 'cpu') -> str:
    """Generate text using the trained model"""
    model.eval()

    # Tokenize start text
    tokens = start_text.split()
    indices = [vocab(token) for token in tokens]

    # Convert to tensor
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        # Generate tokens
        for _ in range(max_length):
            # Get model output
            output = model(input_tensor)

            # Get the last token's logits
            logits = output[0, -1, :]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, 1).item()

            # Add to input tensor
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)

            # Stop if we get an end-of-sequence token
            if next_token == vocab('<EOS>'):
                break

    # Convert back to text
    generated_tokens = input_tensor[0].cpu().tolist()
    generated_text = ' '.join([vocab.idx2word.get(idx, '<UNK>') for idx in generated_tokens])

    return generated_text

def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Calculate BLEU score for text generation (simplified)"""
    # This is a simplified implementation
    # In practice, you'd use nltk or sacrebleu for proper BLEU calculation
    if not predictions or not references:
        return 0.0

    # Simple word overlap metric (not true BLEU)
    total_score = 0.0
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())

        if len(pred_words) == 0 or len(ref_words) == 0:
            continue

        # Calculate Jaccard similarity
        intersection = len(pred_words.intersection(ref_words))
        union = len(pred_words.union(ref_words))
        score = intersection / union if union > 0 else 0.0

        total_score += score

    return total_score / len(predictions) if predictions else 0.0