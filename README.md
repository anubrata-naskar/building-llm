# Building LLM - From Scratch Implementation

A comprehensive project for building and training Large Language Models (LLMs) from scratch using PyTorch. This project implements core LLM components including tokenization, embeddings, self-attention, multi-head attention, and GPT-style transformer models with complete training pipelines.

## 📋 Project Overview

This repository contains Jupyter notebooks that demonstrate the step-by-step implementation of a GPT-style language model, covering everything from basic tokenization to full model training and text generation.

## 🚀 Completed Features

### ✅ **Text Processing & Tokenization** (`text_data.ipynb`)
- **Custom Tokenization**: Regex-based text tokenization with word-level splitting
- **Unknown Token Handling**: Robust handling of out-of-vocabulary tokens with `|unk|` token
- **Multiple Tokenizer Options**:
  - NLTK word tokenization
  - tiktoken (GPT-2 tokenizer)
  - Custom vocabulary from common word lists
- **Token & Positional Embeddings**: PyTorch embedding layers with positional encoding
- **Reusable Components**: Modular embedding classes for easy integration

### ✅ **Attention Mechanisms** (`self_attention.ipynb`, `multi_head_attention.ipynb`)
- **Self-Attention Implementation**: Complete self-attention mechanism with proper matrix dimensions
- **Causal Masking**: Upper triangular masking for autoregressive generation
- **Multi-Head Attention**: Parallel attention heads with concatenation and projection
- **Dynamic Sequence Handling**: Supports variable-length input sequences
- **Attention Analysis**: Tools for visualizing attention weights and patterns

### ✅ **GPT Model Architecture** (`pretraining_unlabeled_data.ipynb`)
- **Complete GPT Implementation**:
  - Token and positional embeddings
  - Multi-layer transformer blocks
  - Causal self-attention
  - Feed-forward networks with GELU activation
  - Layer normalization and residual connections
  - Language modeling head for next-token prediction

### ✅ **Training Infrastructure**
- **Data Loading**: Custom dataset and dataloader for autoregressive language modeling
- **Training Loop**: Complete training pipeline following deep learning best practices:
  - Batch processing with forward/backward passes
  - Gradient calculation and parameter updates
  - Loss tracking for training and validation sets
  - Periodic evaluation and sample generation
- **Optimization**: AdamW optimizer with weight decay
- **Loss Function**: Cross-entropy loss for next-token prediction

### ✅ **Model Analysis & Monitoring**
- **Training Visualization**: Real-time loss curves and overfitting detection
- **Text Generation**: Temperature-controlled sampling for creative text generation
- **Prediction Analysis**: Top-k token prediction analysis
- **Model Statistics**: Parameter counting, memory estimation, architecture summary
- **Checkpointing**: Model saving and loading functionality

## 📊 Model Specifications

### Current Implementation
- **Architecture**: GPT-style transformer
- **Parameters**: ~26.8M (configurable)
- **Layers**: 2 transformer blocks (configurable)
- **Attention Heads**: 4 heads (configurable)
- **Model Dimension**: 256 (configurable)
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Context Length**: 64 tokens (configurable)

### Training Configuration
- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.01)
- **Batch Size**: 8 sequences
- **Sequence Length**: 32 tokens with 16-token stride
- **Epochs**: 3 (demonstrational)
- **Device**: CPU/GPU compatible

## 🔧 Technical Implementation

### Core Components
1. **CausalAttention**: Self-attention with causal masking
2. **MultiHeadCausalAttention**: Multi-head attention mechanism
3. **TransformerBlock**: Complete transformer layer with attention + FFN
4. **GPTModel**: Full GPT architecture with embeddings and language head
5. **GPTDatasetV1**: Custom dataset for autoregressive training

### Key Features
- **Dynamic Sequence Lengths**: Handles variable-length inputs
- **Memory Efficient**: Optimized attention computation
- **Configurable Architecture**: Easy to modify model dimensions
- **Training Monitoring**: Comprehensive logging and visualization
- **Text Generation**: Multiple sampling strategies (greedy, temperature-based)

## 📁 File Structure

```
building-llm/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
├── text_data.ipynb                    # Tokenization and embeddings
├── self_attention.ipynb               # Self-attention implementation
├── multi_head_attention.ipynb         # Multi-head attention
├── pretraining_unlabeled_data.ipynb   # Complete GPT model and training
├── myenv2/                            # Python virtual environment
└── models/                            # Saved model checkpoints
```

## 🛠 Setup & Usage

### Prerequisites
```bash
# Create virtual environment
python -m venv myenv2
myenv2\Scripts\activate  # Windows
source myenv2/bin/activate  # Linux/Mac

# Install dependencies
pip install torch tiktoken matplotlib jupyter nltk
```

### Running the Notebooks
1. **Start with tokenization**: `text_data.ipynb`
2. **Learn attention mechanisms**: `self_attention.ipynb` → `multi_head_attention.ipynb`
3. **Complete model training**: `pretraining_unlabeled_data.ipynb`

Each notebook is designed to be educational and self-contained with detailed explanations.

## 📈 Training Results

### Sample Training Output
```
Epoch 1 (Step 000001): Train loss 10.717, Val loss 10.795
Generated: "The quick leptin Straight Bless Stability..."

Epoch 2: Train loss stable, improving text coherence
Epoch 3: Model checkpoint saved successfully
```

### Model Capabilities
- **Text Generation**: Produces contextually relevant (though not perfectly coherent) text
- **Next-Token Prediction**: Learned basic language patterns
- **Attention Patterns**: Develops meaningful attention weights
- **Scalability**: Architecture supports scaling to larger models

## 🔮 Future Enhancements

### Planned Features
- [ ] **Larger Training Data**: Scale to datasets like OpenWebText
- [ ] **Learning Rate Scheduling**: Implement warmup and decay schedules
- [ ] **Advanced Sampling**: Implement nucleus (top-p) sampling
- [ ] **Model Evaluation**: Perplexity and other language modeling metrics
- [ ] **Fine-tuning**: Task-specific fine-tuning capabilities
- [ ] **Distributed Training**: Multi-GPU training support

### Experimental Ideas
- [ ] **Different Architectures**: RoPE, SwiGLU, RMSNorm
- [ ] **Memory Optimization**: Gradient checkpointing, mixed precision
- [ ] **Advanced Training**: RLHF, instruction tuning
- [ ] **Model Analysis**: Interpretability tools and attention visualization

## 📚 Educational Value

This project serves as a comprehensive tutorial for understanding:
- **Transformer Architecture**: From basic attention to full GPT models
- **Deep Learning Training**: Complete training loops and best practices
- **PyTorch Implementation**: Professional-grade model implementation
- **NLP Fundamentals**: Tokenization, embeddings, and language modeling
- **Model Development**: From prototype to production-ready code

## 🤝 Contributing

This is an educational project. Contributions are welcome for:
- Bug fixes and improvements
- Additional documentation
- New features and experiments
- Performance optimizations

## 📄 License

MIT License - Feel free to use this code for learning and research purposes.

---

**Note**: This implementation is designed for educational purposes and demonstrates the core concepts of building LLMs from scratch. For production use, consider established frameworks like Hugging Face Transformers.
