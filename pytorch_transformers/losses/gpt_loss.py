import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTLoss(nn.Module):
    """Custom loss function for GPT training with label smoothing"""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass through loss function

        Args:
            predictions: Model predictions of shape (batch_size, seq_len, vocab_size)
            targets: Ground truth targets of shape (batch_size, seq_len)

        Returns:
            Scalar loss value
        """
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