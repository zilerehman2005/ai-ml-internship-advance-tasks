# 📰 News Topic Classifier Using BERT

> **AI/ML Engineering Internship — Advanced Task 1**  
> Fine-tune `bert-base-uncased` on the AG News dataset to classify news headlines into topic categories, with a live Gradio demo.

---

## 🎯 Objective

Build a production-ready **multi-class text classification** system that:
1. Fine-tunes BERT on the AG News benchmark dataset
2. Evaluates performance with Accuracy and weighted F1-score
3. Deploys as an interactive Gradio web app for live inference

---

## 📁 Repository Structure

```
bert-news-classifier/
│
├── news_topic_classifier.ipynb   # Main notebook (EDA → Training → Eval → Gradio)
├── app.py                        # Standalone Gradio deployment script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📊 Dataset — AG News

| Property | Value |
|----------|-------|
| Source | [Hugging Face — `ag_news`](https://huggingface.co/datasets/ag_news) |
| Train samples | 120,000 |
| Test samples | 7,600 |
| Classes | 4 (perfectly balanced, 30k / class) |
| Input | News title + description concatenated |

**Label mapping:**
| ID | Category |
|----|----------|
| 0 | 🌍 World |
| 1 | ⚽ Sports |
| 2 | 💼 Business |
| 3 | 🔬 Sci/Tech |

---

## 🧠 Methodology / Approach

### 1. Exploratory Data Analysis
- Visualised class distribution (perfectly balanced — no oversampling needed)
- Analysed word-count distribution per category
- Identified that Business/World have the most semantic overlap

### 2. Tokenisation & Preprocessing
- Used `AutoTokenizer` from `bert-base-uncased`
- Max sequence length: **128 tokens** (covers 99%+ of AG News headlines)
- Dynamic padding via `DataCollatorWithPadding` for efficient batching

### 3. Model Architecture
```
bert-base-uncased
    └── BertModel (12 transformer layers, 768 hidden dim, 12 attention heads)
    └── Dropout(0.1)
    └── Linear(768 → 4)    ← classification head
```
Total parameters: **109,483,012** (all fine-tuned)

### 4. Fine-Tuning Configuration
| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size (train) | 32 |
| Batch size (eval) | 64 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Precision | fp16 (on GPU) |

### 5. Evaluation Metrics
- **Accuracy** — overall correctness
- **Weighted F1** — accounts for any class imbalance; primary selection metric
- **Per-class Precision / Recall / F1** — via `classification_report`
- **Confusion Matrix** — visual analysis of misclassifications

### 6. Deployment
- **Gradio** `gr.Blocks` interface with example headlines and probability bar chart
- Supports both local launch and Hugging Face Spaces hosting

---

## 📈 Key Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~94% |
| Weighted F1 | ~94% |
| Best confused pair | World ↔ Business |

> ✅ Results align with published BERT baselines on AG News (~94.9% accuracy with full training data)

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)
1. Open `news_topic_classifier.ipynb` in Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells top-to-bottom
4. The last cell launches a public Gradio link automatically

### Option B — Local
```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/bert-news-classifier.git
cd bert-news-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook to train & save the model
jupyter notebook news_topic_classifier.ipynb

# 4. Launch the Gradio app
python app.py
```

### Option C — Hugging Face Spaces
1. Create a new Space (Gradio SDK)
2. Upload `app.py`, `requirements.txt`, and the saved model folder `bert-agnews-final/`
3. The Space auto-builds and serves the demo

---

## 🔑 Skills Demonstrated

- ✅ NLP with Transformer models (BERT)
- ✅ Transfer learning & fine-tuning on domain data
- ✅ HuggingFace `datasets`, `transformers`, `evaluate` ecosystem
- ✅ Evaluation: Accuracy, F1, Confusion Matrix, Classification Report
- ✅ Training visualisation (loss curves, metric curves)
- ✅ Lightweight model deployment with Gradio

---

## 🛠️ Possible Improvements

| Idea | Expected Gain |
|------|---------------|
| Use full 120k training set | +0.5–1% accuracy |
| Switch to `roberta-base` | +1–2% accuracy |
| Hyperparameter search (Optuna) | +0.5% F1 |
| Quantisation / ONNX export | 3–4× faster inference |
| Data augmentation | Better generalisation |

---

## 📚 References

- Devlin et al. (2019) — [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- Zhang & LeCun (2015) — [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626) (AG News origin)
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [AG News on HuggingFace Hub](https://huggingface.co/datasets/ag_news)
