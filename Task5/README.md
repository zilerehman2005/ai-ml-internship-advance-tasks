# 🏷️ Auto Tagging Support Tickets Using LLM

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements an automated customer support ticket classification system using three different Large Language Model (LLM) approaches:

1. **Zero-Shot Classification** — No labeled examples; the model classifies based on category names alone
2. **Few-Shot Classification** — A handful of labeled examples are provided in the prompt
3. **Fine-Tuned Classification** — DistilBERT is fine-tuned on the training set for maximum accuracy

All approaches output the **Top 3 most probable tags** with confidence scores, enabling intelligent routing and prioritization of support tickets.

---

## 🎯 Objective

- Automatically assign category labels to incoming support tickets
- Compare Zero-Shot, Few-Shot, and Fine-Tuned LLM approaches
- Evaluate and visualize performance across all methods
- Provide a production-ready classification pipeline

---

## 📊 Dataset Description

| Property | Value |
|---|---|
| **Path** | `/content/customer_support_tickets.csv` |
| **Rows** | 8,469 |
| **Columns** | 17 |
| **Target Column** | `Ticket Type` |
| **Input Features** | `Ticket Subject` + `Ticket Description` |

### Target Categories (extracted automatically)
Examples include: `Technical Issue`, `Billing Inquiry`, `Product Inquiry`, `Cancellation Request`, `Refund Request`, `Account Access`, etc.

---

## 🔬 Methodology

### Feature Selection Justification

| Column | Used? | Reason |
|---|---|---|
| Ticket Subject | ✅ Yes | Core signal for classification |
| Ticket Description | ✅ Yes | Rich semantic content |
| Ticket ID | ❌ No | Identifier only, no signal |
| Customer Name / Email | ❌ No | PII, no signal |
| Resolution | ❌ No | Target leak — unknown at inference time |
| Customer Satisfaction Rating | ❌ No | Post-resolution metric |
| Time to Resolution | ❌ No | Post-resolution metric |

The `combined_text` feature = `Ticket Subject + " " + Ticket Description` captures both the brief summary and full context of a ticket.

---

## 🤖 Approaches

### 1. Zero-Shot Classification (OpenRouter API)
- Uses a frontier LLM via OpenRouter with a carefully crafted prompt
- Lists all possible categories and asks for Top 3 with confidence scores
- No training required — instant deployment
- **Strengths:** Zero setup, generalizes well, interpretable prompts
- **Limitations:** API cost, latency, no domain adaptation

### 2. Few-Shot Classification (OpenRouter API)
- Augments the Zero-Shot prompt with 5–10 labeled examples
- Examples are selected to cover all major categories
- **Strengths:** Improved accuracy over Zero-Shot, still no fine-tuning
- **Limitations:** Prompt length grows with examples, API cost

### 3. Fine-Tuned Classification (DistilBERT)
- `distilbert-base-uncased` fine-tuned on 70% of the dataset
- Uses HuggingFace Trainer API with early stopping
- **Strengths:** Highest accuracy, no API cost at inference, fast
- **Limitations:** Requires training time and GPU, less flexible

---

## 📈 Evaluation Metrics

All methods evaluated on the same held-out test set:

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision (weighted) | Weighted average precision across classes |
| Recall (weighted) | Weighted average recall across classes |
| F1-Score (weighted) | Harmonic mean of Precision and Recall |
| Top-3 Accuracy | Is the correct label in the top 3 predictions? |

---

## 📁 Repository Structure

```
Auto_Tagging_Support_Tickets/
├── Auto_Tagging_Support_Tickets.ipynb   # Main notebook
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── dataset.csv                          # Dataset (add manually)
└── outputs/
    ├── category_distribution.png
    ├── ticket_length_distribution.png
    ├── wordcloud.png
    ├── confusion_matrix_finetuned.png
    ├── accuracy_comparison.png
    ├── f1_comparison.png
    └── predictions.csv
```

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running in Google Colab
1. Upload `customer_support_tickets.csv` to `/content/`
2. Open `Auto_Tagging_Support_Tickets.ipynb` in Google Colab
3. Set your **OpenRouter API key** in the configuration cell
4. Run all cells top to bottom (`Runtime → Run All`)

### OpenRouter API Key
Sign up at [openrouter.ai](https://openrouter.ai) to get a free API key. Set it in the notebook:
```python
OPENROUTER_API_KEY = "your-key-here"
```

---

## 🔮 Future Work

- Deploy as a REST API with FastAPI
- Add active learning for continuous improvement
- Experiment with larger models (GPT-4, Claude-3)
- Implement multi-label classification for overlapping ticket types
- Build a real-time dashboard for ticket monitoring

---

## 👤 Author

Built as part of an NLP Engineering internship project.

---

## 📄 License

MIT License
