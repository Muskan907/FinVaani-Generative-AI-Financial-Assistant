#  FinVaani

### Bilingual Generative AI Financial Assistant

FinVaani is a domain-specific financial question-answering system built for Indian financial literacy. The project fine-tunes **mGPT** using **LoRA (PEFT)** and applies the **Lottery Ticket Hypothesis (LTH)** to compress the model while preserving performance. The system supports both **English and Hindi** queries and is trained on financial resources from **RBI, SEBI, IRDAI, NPCI, and NCFE**.

## Key Highlights

* Fine-tuned multilingual LLM using **LoRA (PEFT)**
* Supports **English and Hindi** financial queries
* Trained on **1,700+ curated financial Q&A pairs**
* Applied **Lottery Ticket Hypothesis** for model compression
* Reduced active parameters by **~80%** while retaining **90%+** performance
* Interactive **Streamlit-based** chat interface

## Tech Stack

**Python • PyTorch • Hugging Face • Transformers • LoRA • PEFT • NLP • Streamlit**

## Workflow

```text
Financial Data Sources
        ↓
 Data Collection
        ↓
   Preprocessing
        ↓
      mGPT
        ↓
 LoRA Fine-Tuning
        ↓
  LTH Pruning
        ↓
    Evaluation
        ↓
  Streamlit App
```

## Results

| Model           | BLEU  | ROUGE-L | Parameters |
| --------------- | ----- | ------- | ---------- |
| Raw mGPT        | 0.040 | 0.120   | 117M       |
| LoRA Fine-Tuned | 0.210 | 0.380   | 118M       |
| Winning Ticket  | 0.190 | 0.350   | ~25M       |

**Outcome:** Achieved significant model compression with minimal performance degradation, enabling faster and more efficient inference.

## Research Concepts

* Generative AI
* Large Language Models (LLMs)
* Parameter-Efficient Fine-Tuning (PEFT)
* LoRA
* Prompt Engineering
* Lottery Ticket Hypothesis
* Model Compression
* Multilingual NLP

## Team

**BML Munjal University**
B.Tech Computer Science Engineering (Data Science & Artificial Intelligence)
