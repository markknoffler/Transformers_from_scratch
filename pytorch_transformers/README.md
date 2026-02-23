# GPT from Scratch

This project implements a complete GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. The implementation follows the core architecture described in the original GPT paper while providing educational clarity and modularity.

## Features

- Complete GPT architecture implementation from scratch
- Causal attention with proper masking
- Multi-head attention mechanism
- Layer normalization and residual connections
- Custom loss function with label smoothing
- Text tokenization and data handling utilities
- Training and evaluation scripts
- Comprehensive documentation

## Architecture Overview

The GPT model is built on a decoder-only transformer architecture with:

1. **Input Embeddings**: Token embeddings + positional embeddings
2. **Transformer Blocks**: Multiple layers of attention + feed-forward networks
3. **Output Layer**: Final projection to vocabulary space
4. **Causal Attention**: Ensures autoregressive behavior

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pytorch_transformers

# Install dependencies
pip install torch numpy
```

## Project Structure

```
pytorch_transformers/
├── models/
│   ├── __init__.py
│   ├── gpt.py          # Main GPT implementation
│   └── gpt_config.py   # Configuration files
├── losses/
│   └── gpt_loss.py     # Custom loss function
├── data/
│   └── data_utils.py   # Data handling utilities
├── scripts/
│   ├── tokenize.py     # Tokenization utilities
│   ├── train.py        # Training script
│   └── evaluate.py     # Evaluation utilities
├── notebooks/
│   └── gpt_exploration.ipynb  # Jupyter notebook for exploration
├── experiments/
│   └── experiment_log.md      # Experimental results and logs
├── docs/
│   └── architecture.md        # Detailed architecture documentation
├── tests/
│   └── test_gpt.py            # Unit tests
└── README.md
```

## Usage

### Training

To train the model on your text data:

```bash
python scripts/train.py \
    --data_path /path/to/your/text/data \
    --save_path ./models \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### Evaluation

The evaluation script provides functions for:
- Perplexity calculation
- Accuracy measurement
- Text generation
- BLEU score computation

### Text Generation

After training, you can generate text using the trained model:

```python
# Load trained model
model = load_model('models/model.pth', vocab, device='cuda')

# Generate text
generated_text = generate_text(model, vocab, "Once upon a time", max_length=100)
print(generated_text)
```

## Model Configuration

The GPT model supports various hyperparameters:

| Parameter | Description | Default |
|----------|-------------|---------|
| `vocab_size` | Size of vocabulary | 1000 |
| `d_model` | Model dimension | 512 |
| `n_heads` | Number of attention heads | 8 |
| `n_layers` | Number of transformer layers | 6 |
| `d_ff` | Feed-forward dimension | 2048 |
| `max_len` | Maximum sequence length | 512 |
| `dropout` | Dropout rate | 0.1 |

## Training Details

### Loss Function

The implementation uses a custom loss function with label smoothing:
- Standard cross-entropy loss
- Label smoothing to prevent overfitting
- Padding token handling

### Optimizer

- AdamW optimizer with weight decay
- Gradient clipping for stability
- Learning rate scheduling (can be extended)

### Evaluation Metrics

- Perplexity calculation
- Token-level accuracy
- Generation quality assessment

## Testing

Run unit tests to verify the implementation:

```bash
python -m unittest tests/test_gpt.py
```

## Experimental Results

The implementation has been tested with various configurations and datasets. Results are documented in `experiments/experiment_log.md`.

## License

MIT License

## Acknowledgments

This implementation is based on the following papers and resources:
- Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
- Radford et al. "Improving Language Understanding by Generative Pre-Training" (OpenAI 2018)
- Brown et al. "Language Models are Few-Shot Learners" (NeurIPS 2020)

## Contributing

Contributions are welcome! Please submit issues and pull requests for improvements or bug fixes.