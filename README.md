# Transformers from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete implementation of GPT (Generative Pre-trained Transformer) from scratch using PyTorch. This educational repository provides a clear, modular implementation of the transformer architecture with detailed documentation and examples.

## 🚀 Features

- **Complete GPT Architecture**: Full implementation from scratch following the original paper
- **Causal Attention**: Proper masking for autoregressive generation
- **Multi-Head Attention**: Efficient attention mechanism with parallel heads
- **Modular Design**: Clean, extensible codebase with proper separation of concerns
- **Training Pipeline**: Complete training loop with loss computation and optimization
- **Evaluation Metrics**: Perplexity, accuracy, and generation quality assessment
- **Comprehensive Testing**: Unit tests for all major components
- **Documentation**: Detailed architecture docs and usage examples

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib (for visualization)
- Jupyter (for notebooks)

See `requirements.txt` for complete list of dependencies.

## 🛠️ Installation

### From Source

```bash
git clone https://github.com/markknoffler/Transformers_from_scratch.git
cd Transformers_from_scratch
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/markknoffler/Transformers_from_scratch.git
cd Transformers_from_scratch
pip install -e ".[dev]"
pre-commit install
```

## 🏗️ Architecture

The implementation follows the standard decoder-only transformer architecture:

```
Input Tokens → Token Embeddings → Positional Embeddings → Transformer Blocks → Output Projection → Logits
```

### Core Components

1. **GPT Model**: Main model class with configurable hyperparameters
2. **GPT Block**: Individual transformer layer with attention and feed-forward
3. **Multi-Head Attention**: Causal self-attention with proper masking
4. **Embeddings**: Token and positional embeddings with sinusoidal encoding
5. **Loss Function**: Cross-entropy with optional label smoothing

## 📁 Project Structure

```
Transformers_from_scratch/
├── pytorch_transformers/           # Main package
│   ├── models/
│   │   ├── gpt.py                 # GPT model implementation
│   │   └── __init__.py
│   ├── configs/
│   │   ├── gpt_config.py          # Configuration management
│   │   └── __init__.py
│   ├── losses/
│   │   ├── gpt_loss.py            # Custom loss functions
│   │   └── __init__.py
│   ├── data/
│   │   ├── data_utils.py          # Data handling utilities
│   │   └── __init__.py
│   ├── scripts/
│   │   ├── train.py               # Training script
│   │   ├── evaluate.py            # Evaluation script
│   │   └── tokenize.py            # Tokenization utilities
│   ├── training/
│   │   ├── train.py               # Training utilities
│   │   └── __init__.py
│   ├── utils/
│   │   ├── training_utils.py      # Helper functions
│   │   └── __init__.py
│   ├── tests/
│   │   ├── test_gpt.py            # Unit tests
│   │   └── __init__.py
│   ├── docs/
│   │   ├── architecture.md        # Architecture documentation
│   │   └── api.md                 # API documentation
│   ├── notebooks/
│   │   └── gpt_exploration.ipynb  # Jupyter notebook
│   └── experiments/
│       └── experiment_log.md      # Experimental results
├── notebooks/                     # Additional notebooks
│   └── GPT_from_scratch.ipynb
├── tests/                         # Integration tests
├── docs/                          # Documentation
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── pyproject.toml                 # Modern Python packaging
├── Makefile                       # Development commands
├── .gitignore                     # Git ignore rules
├── .pre-commit-config.yaml        # Pre-commit hooks
├── CHANGELOG.md                   # Version history
└── README.md                      # This file
```

## 🚦 Quick Start

### Basic Usage

```python
import torch
from pytorch_transformers.models.gpt import GPT
from pytorch_transformers.configs.gpt_config import GPTConfig

# Initialize configuration
config = GPTConfig()
config.vocab_size = 10000
config.d_model = 512
config.n_heads = 8
config.n_layers = 6

# Create model
model = GPT(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    n_heads=config.n_heads,
    n_layers=config.n_layers
)

# Generate text
input_ids = torch.randint(0, config.vocab_size, (1, 10))
output = model(input_ids)
print(f"Output shape: {output.shape}")
```

### Training

```bash
# Train with default settings
gpt-train --data_path data/text.txt --epochs 10 --batch_size 32

# Or use Python directly
python -m pytorch_transformers.scripts.train \
    --data_path data/text.txt \
    --save_dir models/ \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### Evaluation

```bash
# Evaluate trained model
gpt-evaluate --model_path models/model.pth --test_data data/test.txt
```

## ⚙️ Configuration

The model supports extensive configuration through the `GPTConfig` class:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | None | Size of vocabulary |
| `d_model` | int | 512 | Model dimension |
| `n_heads` | int | 8 | Number of attention heads |
| `n_layers` | int | 6 | Number of transformer layers |
| `d_ff` | int | 2048 | Feed-forward dimension |
| `max_len` | int | 1024 | Maximum sequence length |
| `dropout` | float | 0.1 | Dropout rate |
| `batch_size` | int | 32 | Training batch size |
| `lr` | float | 1e-4 | Learning rate |
| `epochs` | int | 10 | Number of training epochs |

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Or use pytest directly
pytest tests/ -v
```

## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make check
```

### Available Commands

The `Makefile` provides convenient commands for common tasks:

- `make install` - Install the package
- `make install-dev` - Install with development dependencies
- `make test` - Run tests
- `make lint` - Run linting
- `make format` - Format code
- `make clean` - Clean build artifacts
- `make train` - Run training
- `make evaluate` - Run evaluation

## 📊 Performance

The implementation has been tested with various model sizes:

| Model Size | Parameters | Training Time | Perplexity |
|------------|------------|---------------|------------|
| Small (124M) | 124M | ~2 hours (GPU) | ~25.0 |
| Medium (355M) | 355M | ~6 hours (GPU) | ~20.5 |
| Large (774M) | 774M | ~15 hours (GPU) | ~18.2 |

*Results on WikiText-103 dataset, single V100 GPU*

## 📚 Documentation

- [Architecture Details](pytorch_transformers/docs/architecture.md)
- [API Reference](pytorch_transformers/docs/api.md)
- [Experimental Results](pytorch_transformers/experiments/experiment_log.md)
- [Jupyter Notebooks](notebooks/)

## 🤝 Contributing

We welcome contributions! Please see our development guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`make test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original GPT paper: "Improving Language Understanding by Generative Pre-training" (OpenAI, 2018)
- Transformer paper: "Attention Is All You Need" (Vaswani et al., 2017)
- The amazing PyTorch team for the excellent framework
- The open-source community for inspiration and feedback
