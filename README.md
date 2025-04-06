# 🇻🇳 ViLM: Vietnamese Language Model

**ViLM (Vietnamese Language Model)** is an end-to-end pipeline for building, training, and deploying Transformer-based language models specifically tailored for the Vietnamese language.

This repository provides the components to:

- 🔤 **Build** a custom tokenizer for Vietnamese
- 🧠 **Train** modern Transformer-based models
- ⚡ **Optimize and deploy** models with a high-performance C++ engine

---

## 📌 Project Structure

```
vilm/
├── tokenizer/      # Tokenizer building, training, and evaluation tools
│   ├── data/           # Sample Vietnamese text for tokenizer training
│   └── ...             # Scripts (e.g., train_tokenizer.py)
├── model/          # Model architecture definitions and training scripts
│   ├── configs/        # YAML configuration files for training runs
│   └── ...             # Training scripts, model implementations
├── inference/      # Optimized inference code and tools
│   ├── c_runtime/      # C/C++ based inference engine
│   └── ...             # Optimization scripts, benchmarking tools
├── data/           # (Optional) Larger Vietnamese corpora for model pre-training
└── LICENSE         # Project license file
```

---

## 🔤 1. [Build Tokenizer](https://github.com/vietnlp/vilm/tree/main/tokenizer)

We developed a custom tokenizer designed specifically for Vietnamese, addressing issues common with general-purpose tokenizers when applied to this language.

### 🚫 Problems with Existing Tokenizers

- Split common Vietnamese words into 3–4 tokens
- Produce fragmented or meaningless subwords
- Poor handling of numerals, scientific terms, and special characters
- Include irrelevant foreign words in the vocabulary
- Lack optimization for Vietnamese grammar and syntax

### ✅ Advantages of Our Tokenizer

- **Reduced fragmentation** for better word representation
- **Fewer tokens per sentence**, improving model efficiency
- **Improved semantic accuracy** through meaningful tokens
- **Localized vocabulary** for Vietnamese language and culture

### ✨ Features

- **Collect & Analyze Corpora**
  - Curate high-quality Vietnamese datasets
  - Clean, deduplicate, and normalize text

- **Tokenizer Benchmarking**
  - Evaluate popular tokenizers:
    - General-purpose: GPT-4o, LLaMA 3, Gemma 2, Qwen 2.5
    - Vietnamese-specific: **PhoGPT**
  - Metrics: token count, vocab coverage, semantic coherence

- **Compare Tokenization Algorithms**
  - Subword methods supported:
    - BPE (Byte-Pair Encoding)
    - Unigram Language Model
    - WordPiece
    - Character-level and hybrid variants

- **Train Custom Tokenizer**
  - Based on [SentencePiece](https://github.com/google/sentencepiece)
  - Choose from `bpe`, `unigram`, etc.
  - Evaluate vocab size, token efficiency, and downstream performance

- **Vietnamese-specific Preprocessing**
  - Rule-based normalization for accents, compound words, etc.
  - Intelligent handling of abbreviations and scientific terms

- **Export Artifacts**
  - Save tokenizer model and configuration for reuse in training or inference

### 📚 References

- [Pretraining LLMs – DeepLearning.AI](https://www.deeplearning.ai/short-courses/pretraining-llms/)
- [The Role of Tokenizers – DeepLearning.AI](https://www.deeplearning.ai/short-courses/retrieval-optimization-from-tokenization-to-vector-quantization/)

### ▶️ How to Use

```bash
cd tokenizer
python train_tokenizer.py --input data/vietnamese_corpus.txt --model_type bpe --vocab_size 32000
```

---

## 🧠 2. [Build Model](https://github.com/vietnlp/vilm/tree/main/model)

This module handles model training using PyTorch. It supports baseline and experimental architectures for training from scratch on Vietnamese text.

### 🧱 Supported Architectures & Features

- Standard Decoder-Only Transformer
- Mixture-of-Experts (MoE) layers
- Mamba (State Space Models)
- Multi-head & Grouped-Query Attention (like LLaMA)
- Multi-latent attention (inspired by DeepSeek)
- Experimental: Diffusion-based LLMs

### 🛠️ Training Workflow

- Scripts and configs for pretraining on large Vietnamese corpora
- Easily configurable architecture, hyperparameters, and training schemes
- Designed for extensibility and research

---

## ⚡ 3. [Inference](https://github.com/vietnlp/vilm/tree/main/inference)

This module optimizes and deploys trained models for fast, memory-efficient inference.

### 🚀 Optimization Techniques

- **FlashAttention** – Accelerated attention computation
- **Quantization** – Use AWQ to reduce memory usage with minimal accuracy loss
- **Knowledge Distillation** – Smaller models mimic larger ones
- **Pruning** – Remove redundant weights

### ⚙️ C++ Inference Engine

- Convert PyTorch models to C++ runtime using:
  - `llama.cpp`
  - `GGML`
  - Custom lightweight engines
- Target: ultra-low-latency deployment and edge compatibility

### 📊 Benchmarking

- Tools to measure model performance: latency, memory use, and throughput

---

## 🤝 Contributing

We welcome contributions from the community! You can:

- 🐛 Fix bugs or improve code
- 🌟 Add new features (e.g., new model types, tokenizer improvements)
- ⚙️ Enhance training or inference performance

Please open an issue or submit a PR to get involved.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for full details.

---

