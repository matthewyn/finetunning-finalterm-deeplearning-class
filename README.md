# Fine-tuning Transformer Models - Deep Learning Final Project

## 📋 Project Overview

This repository contains the complete implementation of three transformer model fine-tuning tasks for the Deep Learning course final project (UAS). The project demonstrates proficiency in working with different transformer architectures: Encoder (DistilBERT), Encoder-Decoder (T5), and Decoder-only (Phi-2) models.

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

### Task 1: DistilBERT for Text Classification 🏷️

**Directory:** `task-1/notebooks/`

**Notebooks:**

- `finetuning_distilbert_text_classification.ipynb` - General text classification task
- `finetuning_distilbert_go_emotions.ipynb` - Emotion detection (GoEmotions dataset)
- `finetuning_distilbert_nli.ipynb` - Natural Language Inference (MNLI dataset)

**Details:**

- **Model:** DistilBERT
- **Architecture:** Encoder-only
- **Task Type:** Sequence Classification
- **Expected Accuracy:** 91-95% (depending on dataset)
- **Training Time:** 10 min - 15 min

**Key Features:**

- Multi-class and multi-label classification
- Bidirectional context understanding
- Transfer learning from pre-trained BERT
- Evaluation with accuracy, precision, recall, F1

---

### Task 2: T5 for Question Answering ❓

**Directory:** `Task 2 (T5 QA on SQuAD)/`

**Notebook:**

- `T5_QA_on_SQuAD.ipynb` - Question answering on SQuAD dataset

**Details:**

- **Model:** T5-base
- **Architecture:** Encoder-Decoder (Seq2Seq)
- **Dataset:** SQuAD v1.1
- **Task Type:** Extractive QA (Generative)
- **Expected F1:** 85-88%
- **Training Time:** 20-25 min

**Key Features:**

- Text-to-text unified framework
- Answer span generation
- Beam search for better quality
- Evaluation with Exact Match and F1 score

---

### Task 3: Phi-2 for Text Summarization 📝

**Directory:** `task-3/notebooks/`

**Notebook:**

- `finetuning_phi_2_text_summarization.ipynb` - Text summarization task

**Details:**

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
finalterm-dl/
├── README.md (this file)
├── requirements.txt
├── task-1/
│   └── notebooks/
│       ├── finetuning_distilbert_go_emotions.ipynb
│       ├── finetuning_distilbert_nli.ipynb
│       └── finetuning_distilbert_text_classification.ipynb
├── Task 2 (T5 QA on SQuAD)/
│   └── T5_QA_on_SQuAD.ipynb
└── task-3/
    └── notebooks/
        └── finetuning_phi_2_text_summarization.ipynb
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
cd finetuning_distilbert_text_classification && pip install -r requirements.txt
cd T5_QA_on_SQuAD && pip install -r requirements.txt
cd finetuning_phi_2_text_summarization && pip install -r requirements.txt
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

- Task 1: `finetuning-distilbert-text-classification`, `finetuning_distilbert_nli`, `finetuning_distilbert_go_emotions`
- Task 2: `T5_QA_on_SQuAD`
- Task 3: `finetuning-phi-2-text-summarization`

## 🌟 Project Highlights

- ✅ Three different transformer architectures implemented
- ✅ Multiple datasets and task types covered
- ✅ State-of-the-art techniques (LoRA, quantization)
- ✅ Comprehensive documentation and analysis
- ✅ Reproducible results with clear instructions
- ✅ Memory-efficient implementations for consumer GPUs
