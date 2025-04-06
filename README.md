# ViLM

ViLM (Vietnamese Language Model) is an end-to-end pipeline for building and deploying a Transformer-based language model for Vietnamese. This repository includes all the necessary components to:

- **Build a tokenizer from scratch**
- **Train a standard Transformer-based language model**
- **Run optimized inference using C for deployment speed**

The project is split into three main parts:

---

## 🔤 1. [Build Tokenizer](https://github.com/vietnlp/vilm/tree/main/tokenizer)

We use a subword tokenizer based on [SentencePiece](https://github.com/google/sentencepiece) to efficiently tokenize Vietnamese text and prepare it for language model training.

### ✨ Features

- **Collect & Analyze Vietnamese Corpora**
  - Identify and process high-quality Vietnamese text datasets for training.
  - Perform cleaning, deduplication, and filtering tailored for Vietnamese.

- **Tokenizer Model Benchmarking**
  - Evaluate the performance of popular tokenizer models on Vietnamese:
    - GPT-4o, LLaMA 3, Gemma, Qwen 2.5
    - Vietnamese-specific models like **PhoGPT**
  - Compare tokenization quality, vocabulary coverage, and efficiency.

- **Compare Tokenizer Algorithms**
  - Study and benchmark different subword tokenization methods:
    - BPE (Byte-Pair Encoding)
    - Unigram Language Model
    - WordPiece
    - Others (character-level, hybrid approaches)

- **Train Your Own Tokenizer**
  - Use SentencePiece to train a Vietnamese tokenizer from scratch.
  - Choose models.
  - Analyze performance: vocabulary size, token efficiency, and training speed.

- **Vietnamese-specific Preprocessing**
  - Apply rule-based normalization and text cleaning tailored for Vietnamese language characteristics (accents, compound words, etc.).

- **Export Tokenizer Artifacts**
  - Export trained tokenizer vocabulary and model configs for model training and downstream inference.

---

### How to use:
```bash
cd tokenizer
python train_tokenizer.py --input data/vietnamese_corpus.txt --model_type bpe --vocab_size 32000
```

---

## 🧠 2. [Build Model](https://github.com/vietnlp/vilm/tree/main/model)

This component trains a Transformer-based Language Model (like GPT-style or BERT-style) from scratch using PyTorch.

### Features:
- Transformer encoder or decoder architecture.
- Custom training loop with logging and checkpointing.
- Supports multi-GPU and mixed precision training (via PyTorch Lightning or native AMP).

### How to train:
```bash
cd model
python train.py --config configs/train_config.yaml
```

---

## ⚡ 3. [Inference](https://github.com/vietnlp/vilm/tree/main/inference)

This module converts trained models into an optimized format and runs inference using C/C++ for maximum performance.

### Features:
- Convert PyTorch model to ONNX or a custom format.
- C-based runtime for fast and efficient inference.
- Benchmarking tools and latency profiling.

### How to run:
```bash
cd inference
make
./vilm_infer --model_path model.bin --input_text "Xin chào thế giới"
```

---

## 📦 Installation

You’ll need Python 3.8+, PyTorch, and CMake for building the inference engine.

```bash
# Python dependencies
pip install -r requirements.txt

# Build C++ inference engine
cd inference
mkdir build && cd build
cmake ..
make
```

---

## 📁 Folder Structure

```
vilm/
├── tokenizer/      # Tokenizer training and utilities
├── model/          # Model training code
├── inference/      # Optimized inference in C
├── data/           # (Optional) Sample Vietnamese corpora
└── configs/        # YAML configs for training
```

---

## 🤝 Contribution

We welcome contributions from the community! You can:
- Fix bugs or issues
- Add new features (e.g., model types, data augmentation)
- Improve inference or training speed

Please open an issue or PR to get started.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Would you like help writing a sample config file or adding more advanced usage tips like distributed training or C profiling?