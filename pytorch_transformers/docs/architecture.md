# GPT Architecture Documentation

## Overview

This document provides a comprehensive technical overview of the GPT architecture implemented from scratch. The implementation follows the core principles of the original GPT paper while maintaining a clean, educational codebase structure.

## Core Architecture Components

### 1. Model Structure

The GPT model is a decoder-only transformer architecture that consists of:

1. **Input Embeddings**: Token embeddings + positional embeddings
2. **Transformer Blocks**: Multiple layers of attention + feed-forward networks
3. **Output Layer**: Final projection to vocabulary space

### 2. Mathematical Foundation

#### Multi-Head Attention

The attention mechanism is defined as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (Query) = $XW^Q$
- $K$ (Key) = $XW^K$
- $V$ (Value) = $XW^V$
- $d_k$ = dimension of keys

For multi-head attention:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### Causal Attention

In GPT, we use causal (masked) attention where each token can only attend to previous tokens:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V$$

Where the mask is:
$$\text{mask}_{i,j} = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}$$

#### Feed-Forward Network

$$\text{FFN}(x) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(x)))$$

#### Positional Encoding

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

#### Layer Normalization

$$\text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

#### Residual Connections

$$\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

## Implementation Details

### 1. GPT Model Class

The main `GPT` class implements the complete architecture:

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        # Embeddings
        self.embedding = GPTEmbeddings(vocab_size, d_model, max_len, dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
```

### 2. Embeddings Module

The `GPTEmbeddings` class combines token and positional embeddings:

```python
class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
```

### 3. Transformer Block

Each `GPTBlock` contains:
- Multi-head attention with causal masking
- Feed-forward network
- Layer normalization and residual connections

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
```

### 4. Loss Function

The custom `GPTLoss` implements label smoothing for better generalization:

```python
class GPTLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
```

## Training Process

### 1. Data Preparation

The training process involves:
- Tokenizing text data
- Creating input-output pairs (context → next token)
- Handling padding and masking

### 2. Forward Pass

1. Input tokens are embedded
2. Positional information is added
3. Transformer blocks process the sequence with causal attention
4. Final projection maps to vocabulary space

### 3. Backward Pass

1. Loss is calculated using cross-entropy
2. Gradients are computed
3. Model parameters are updated using AdamW optimizer

## Key Features

### 1. Causal Masking

The implementation ensures proper autoregressive behavior:
- Each token only attends to previous tokens
- Prevents information leakage from future tokens
- Enables text generation capability

### 2. Layer Normalization

Stabilizes training through:
- Normalizing activations across features
- Reducing internal covariate shift
- Improving gradient flow

### 3. Residual Connections

Enables:
- Deeper networks without vanishing gradients
- Better information flow
- Improved training stability

### 4. Label Smoothing

Improves generalization by:
- Preventing overconfidence in predictions
- Encouraging better probability distributions
- Reducing overfitting

## Configuration Parameters

### Model Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Size of vocabulary | 1000 |
| `d_model` | Model dimension | 512 |
| `n_heads` | Number of attention heads | 8 |
| `n_layers` | Number of transformer layers | 6 |
| `d_ff` | Feed-forward dimension | 2048 |
| `max_len` | Maximum sequence length | 512 |
| `dropout` | Dropout rate | 0.1 |

### Training Hyperparameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `batch_size` | Training batch size | 32 |
| `learning_rate` | AdamW learning rate | 1e-4 |
| `label_smoothing` | Loss function smoothing | 0.1 |
| `epochs` | Training epochs | 10 |

## Performance Characteristics

### Memory Usage

The implementation is designed for efficient memory usage:
- Batch processing for GPU optimization
- Proper tensor shapes and types
- Memory-efficient attention computation

### Training Stability

The architecture includes features for stable training:
- Proper initialization
- Layer normalization
- Residual connections
- Label smoothing

### Scalability

The design supports:
- Easy parameter tuning
- Configurable model sizes
- Efficient implementation
- Modular code structure

## Mathematical Properties

### Attention Properties

1. **Self-Attention**: Each token attends to all tokens in the sequence
2. **Causal Property**: Future tokens are masked to prevent leakage
3. **Multi-Head**: Parallel attention mechanisms for different representations
4. **Scaled Dot-Product**: Proper scaling prevents softmax saturation

### Model Properties

1. **Expressiveness**: Can model long-range dependencies
2. **Parallelization**: Attention computation can be parallelized
3. **Modularity**: Each block is independent
4. **Scalability**: Performance scales with model size

## Optimization Techniques

### 1. Weight Initialization

Xavier initialization for linear layers:
- Ensures proper gradient flow
- Prevents vanishing/exploding gradients
- Maintains stable training

### 2. Layer Normalization

Applied after:
- Attention mechanisms
- Feed-forward networks
- Residual connections

### 3. Dropout

Applied in:
- Attention mechanisms
- Feed-forward networks
- Embeddings

## Usage Examples

### Model Creation

```python
model = GPT(
    vocab_size=1000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_len=512
)
```

### Training Loop

```python
# Forward pass
output = model(input_ids)

# Loss calculation
loss = criterion(output, targets)

# Backward pass
loss.backward()
optimizer.step()
```

### Text Generation

```python
# Generate text
generated = model.generate(
    input_text,
    max_length=100,
    temperature=0.8
)
```

## Future Improvements

### 1. Advanced Features

- Rotary Positional Embeddings
- ALiBi (Attention with Linear Biases)
- Sparse Attention
- Flash Attention

### 2. Training Enhancements

- Gradient checkpointing
- Mixed precision training
- Advanced optimizers
- Learning rate scheduling

### 3. Architecture Extensions

- GPT-2 style improvements
- GPT-3 scale considerations
- Better initialization techniques
- Enhanced regularization

## References

1. Vaswani, A., et al. "Attention Is All You Need." *NeurIPS* 2017.
2. Radford, A., et al. "Improving Language Understanding by Generative Pre-Training." *OpenAI* 2018.
3. Brown, T., et al. "Language Models are Few-Shot Learners." *NeurIPS* 2020.
4. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." *ACL* 2019.