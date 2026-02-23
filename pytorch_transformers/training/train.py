import torch
import torch.nn as nn
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt import GPT
from losses.gpt_loss import GPTLoss
from data.data_utils import Vocab, load_data
from utils.training_utils import train_epoch, validate_epoch, save_checkpoint
import argparse
import os

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GPT Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed forward dimension')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup steps for learning rate')
    parser.add_argument('--print_every', type=int, default=100, help='Print every N batches')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create vocabulary
    vocab = Vocab()

    # Load data
    print("Loading data...")
    texts = load_data(args.data_path)

    # Add all unique words to vocabulary
    for text in texts:
        tokens = text.split()
        for token in tokens:
            vocab.add_token(token)

    print(f"Vocabulary size: {len(vocab)}")

    # Initialize model
    print("Initializing model...")
    model = GPT(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len
    )

    # Move model to device
    device = torch.device(args.device)
    model = model.to(device)

    # Initialize loss function and optimizer
    loss_fn = GPTLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Simple learning rate scheduler (you could implement a more sophisticated one)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # For simplicity, we'll use a basic approach here
        # In a real implementation, you'd want to create a proper DataLoader
        # For now, let's just show the concept

        # This is a simplified version - in practice, you'd create a proper dataset
        # and DataLoader for the GPT architecture
        print("Note: This is a simplified training example.")
        print("In a full implementation, you'd create a proper DataLoader")
        print("with sequence-to-sequence batches for GPT training.")

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch + 1, 0.0, checkpoint_path)

    print("Training completed!")

if __name__ == "__main__":
    main()