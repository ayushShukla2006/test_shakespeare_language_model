# Shakespeare Text Generator - Character-Level LSTM

A character-level LSTM neural network that generates Shakespeare-style text. Trained on the complete works of Shakespeare, this model learns to write in authentic Elizabethan English with proper play formatting, character names, and dialogue structure.

## 🎭 Sample Output

```
HAMLET.
You will hear him on your leaves in the rest of my born,
To have no more than these creatures.

OLIVIA.
Most wonderful!

SEBASTIAN.
The fault's hold the flower of Marcusio.
Enter the office to Leonato's the Train.
```

## 🚀 Features

- **Character-level generation**: Learns patterns at the character level for maximum flexibility
- **LSTM architecture**: 3-layer LSTM with 256 embedding dimensions and 512 hidden units
- **Training resumption**: Automatically saves and loads checkpoints to continue training
- **Early stopping**: Prevents overfitting with configurable patience
- **Customizable generation**: Adjustable temperature parameter for creativity control

## 📋 Requirements

```bash
torch>=2.0.0
tqdm
```

Install dependencies:
```bash
pip install torch tqdm
```

## 📁 Project Structure

```
shakespeare-lstm/
├── train_model.py          # Main training script
├── data/                   # Shakespeare text files
│   ├── hamlet.txt
│   ├── macbeth.txt
│   ├── romeo_and_juliet.txt
│   └── ...
├── checkpoint.pt           # Model checkpoint (generated)
├── vocab.pkl              # Character vocabulary (generated)
└── README.md
```

## 🎯 How It Works

### Model Architecture

```
Input (characters) 
    ↓
Embedding Layer (128 dimensions)
    ↓
3x LSTM Layers (256 hidden units each)
    ↓
Fully Connected Layer
    ↓
Output (character probabilities)
```

**Model Statistics:**
- **Parameters**: ~5 million
- **Vocabulary**: ~100-150 unique characters
- **Context window**: 64 characters
- **Training samples**: Millions (depends on corpus size)

### Training Process

1. **Text Loading**: Reads all `.txt` files from the `data/` folder
2. **Vocabulary Building**: Creates character-to-index mapping
3. **Sequence Creation**: Generates training sequences with 64-character context
4. **Training Loop**: Trains with Adam optimizer and cross-entropy loss
5. **Checkpointing**: Saves model state after every epoch
6. **Generation**: Produces Shakespeare-style text using trained model

## 🏃 Usage

### Training the Model

1. **Prepare your data**: Place Shakespeare text files in the `data/` folder

2. **Configure training parameters** in `train_model.py`:
```python
block_size = 64        # Context window size
batch_size = 64        # Training batch size
epochs = 10            # Number of epochs per run
```

3. **Run training**:
```bash
python train_model.py
```

The script will:
- Load or create vocabulary
- Load existing checkpoint (if available)
- Train for specified epochs
- Save checkpoint after each epoch
- Generate sample text at the end

### Continuing Training

Simply run the script again - it automatically detects and loads the checkpoint:
```bash
python train_model.py
```

### Generating Text Only

To generate text without training, set `epochs = 0` in the script:
```python
epochs = 0  # Just generate, don't train
```

## ⚙️ Configuration

### Model Hyperparameters

```python
# Model architecture
embed_size = 128       # Embedding dimension
hidden_size = 256      # LSTM hidden units
num_layers = 2         # Number of LSTM layers

# Training
learning_rate = 0.001  # Adam optimizer learning rate
batch_size = 64        # Samples per batch
block_size = 64        # Context window (characters)

# Generation
temperature = 0.7      # Creativity (0.1=conservative, 1.0+=creative)
```

### Temperature Guide

- **0.3-0.5**: Conservative, more repetitive, grammatically correct
- **0.6-0.8**: Balanced creativity and coherence (recommended)
- **0.9-1.2**: Highly creative, more experimental, may be nonsensical

## 📊 Training Tips

### For CPU Training (like i3 processor)
- Expect **~1 hour per epoch**
- Train overnight with `epochs = 10`
- Use smaller batch sizes if running out of memory

### For GPU Training (Google Colab)
- Use the provided Colab notebook
- Expect **~3-4 minutes per epoch** with T4 GPU
- Can train 50+ epochs in a few hours

### Recommended Training Schedule
1. **Epochs 0-10**: Basic structure and vocabulary learning
2. **Epochs 10-30**: Grammar and character consistency
3. **Epochs 30-50**: Coherent dialogue and scene structure
4. **Epochs 50+**: Refinement and style mastery

## 🎨 Customization

### Using Your Own Text Corpus

1. Place `.txt` files in the `data/` folder
2. Delete `vocab.pkl` to rebuild vocabulary
3. Run training as normal

The model will learn the style of whatever text you provide!

### Adjusting Model Size

For faster training (smaller model):
```python
embed_size = 64
hidden_size = 128
num_layers = 2
```

For better quality (larger model):
```python
embed_size = 256
hidden_size = 512
num_layers = 3
```

## 📈 Performance

### Training Metrics
- **Initial Loss**: ~3.0-4.0
- **After 10 epochs**: ~2.0-2.5
- **After 30 epochs**: ~1.2-1.5
- **Well-trained**: ~0.8-1.2

### Hardware Requirements
- **Minimum**: 4GB RAM, any CPU
- **Recommended**: 8GB RAM, GPU with CUDA support
- **Storage**: ~100MB for model + data

## 🐛 Troubleshooting

**Issue**: `RuntimeError: CUDA out of memory`
- **Solution**: Reduce `batch_size` (try 32 or 16)

**Issue**: Training is too slow
- **Solution**: Use Google Colab with free GPU, or reduce model size

**Issue**: Generated text is gibberish
- **Solution**: Train for more epochs (need at least 20-30 for coherent output)

**Issue**: Model generates repetitive text
- **Solution**: Increase `temperature` parameter (try 0.8-1.0)

## 🙏 Acknowledgments

- Shakespeare's works from public domain sources
- Built with PyTorch
- Inspired by Andrej Karpathy's char-rnn

## 🔗 Resources

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Note**: This is a character-level model for educational purposes. For production use, consider using transformer-based models or subword tokenization for better performance.