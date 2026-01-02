# Fine-tuning Transformer Models - Deep Learning Final Project

## 📋 Project Overview

This repository contains the complete implementation of three transformer model fine-tuning tasks for the Deep Learning course final project (UAS). The project demonstrates proficiency in working with different transformer architectures: Encoder (BERT), Encoder-Decoder (T5), and Decoder-only (Phi-2) models.

## 👥 Team Information

- **Name:** Kenneth Matthew Yonathan, Diedrick Darrell Darmadi
- **Class:** TK-46-02
- **NIM:** 1103220070, 1103223031

## 🎯 Project Objectives

Master end-to-end deep learning pipelines by fine-tuning state-of-the-art transformer models on different NLP tasks:

1. **Text Classification** using encoder models
2. **Question Answering** using encoder-decoder models
3. **Text Summarization** using decoder-only models

## 📚 Tasks Overview

### Task 1: BERT for Text Classification 🏷️

**Repository:** `task-1`

- **Model:** DistilBERT
- **Architecture:** Encoder-only
- **Dataset Options:** AG News, GoEmotions, MNLI
- **Task Type:** Sequence Classification
- **Expected Accuracy:** 91-95% (depending on dataset and model)
- **Training Time:** 30 min - 2 hours

**Key Features:**

- Multi-class and multi-label classification
- Bidirectional context understanding
- Transfer learning from pre-trained BERT
- Evaluation with accuracy, precision, recall, F1

---

### Task 2: T5 for Question Answering ❓

**Repository:** `task-2`

- **Model:** T5-base
- **Architecture:** Encoder-Decoder (Seq2Seq)
- **Dataset:** SQuAD v1.1
- **Task Type:** Extractive QA (Generative)
- **Expected F1:** 85-88%
- **Training Time:** 2-3 hours

**Key Features:**

- Text-to-text unified framework
- Answer span generation
- Beam search for better quality
- Evaluation with Exact Match and F1 score

---

### Task 3: Phi-2 for Text Summarization 📝

**Repository:** `task-3`

- **Model:** Phi-2 (2.7B parameters)
- **Architecture:** Decoder-only (LLM)
- **Dataset:** XSum
- **Task Type:** Abstractive Summarization
- **Expected ROUGE-L:** 28-32%
- **Training Time:** 20-30 minutes (with optimizations)

**Key Features:**

- Parameter-efficient fine-tuning (LoRA)
- 4-bit quantization for memory efficiency
- Instruction-style prompting
- Evaluation with ROUGE metrics

---

## 🗂️ Repository Structure

```
deep-learning-final-project/
├── README.md (this file)
├── requirements.txt
├── task-1/
│   ├── notebooks/
│   ├── reports/
├── task-2/
│   ├── notebooks/
│   ├── reports/
└── task-3/
    ├── notebooks/
    ├── reports/
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/matthewyn/finetunning-finalterm-deeplearning-class.git
cd deep-learning-final-project
```

### 2. Install Dependencies

```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or install per task
cd finetuning-bert-text-classification && pip install -r requirements.txt
cd finetuning-t5-question-answering && pip install -r requirements.txt
cd finetuning-phi-2-text-summarization && pip install -r requirements.txt
```

### 3. Run Tasks

Each task has detailed instructions in its own README. General workflow:

```bash
# Navigate to task directory
cd finetuning-[task-name]

# Run Jupyter notebook
jupyter notebook notebooks/

# Or run Python script
python src/train.py
```

## 📊 Results Summary

| Task               | Model      | Dataset | Metric   | Score |
| ------------------ | ---------- | ------- | -------- | ----- |
| Classification     | BERT       | AG News | Accuracy | 94.5% |
| Classification     | DistilBERT | AG News | Accuracy | 93.2% |
| Question Answering | T5-base    | SQuAD   | F1       | 86.7% |
| Question Answering | T5-base    | SQuAD   | EM       | 79.8% |
| Summarization      | Phi-2      | XSum    | ROUGE-L  | 30.5% |
| Summarization      | Phi-2      | XSum    | ROUGE-1  | 36.8% |

## 🔧 Technologies Used

### Core Frameworks

- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace transformers library
- **Datasets** - HuggingFace datasets library

### Optimization

- **PEFT** - Parameter-Efficient Fine-Tuning (LoRA)
- **bitsandbytes** - 4-bit and 8-bit quantization
- **Accelerate** - Distributed training support

### Evaluation

- **evaluate** - HuggingFace evaluation library
- **scikit-learn** - ML metrics
- **rouge-score** - ROUGE metrics for summarization

### Visualization & Analysis

- **matplotlib** - Plotting
- **seaborn** - Statistical visualizations
- **pandas** - Data analysis

## 🎯 Submission Guidelines

### Repository Naming

- Task 1: `finetuning-distilbert-text-classification`, `finetuning_distilbert_nli`
- Task 2: `finetuning-t5-question-answering`
- Task 3: `finetuning-phi-2-text-summarization`

## 🌟 Project Highlights

- ✅ Three different transformer architectures implemented
- ✅ Multiple datasets and task types covered
- ✅ State-of-the-art techniques (LoRA, quantization)
- ✅ Comprehensive documentation and analysis
- ✅ Reproducible results with clear instructions
- ✅ Memory-efficient implementations for consumer GPUs
