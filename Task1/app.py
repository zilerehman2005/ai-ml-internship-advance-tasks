"""
app.py — Gradio deployment for the BERT News Topic Classifier.

Run locally:
    python app.py

Run on Hugging Face Spaces:
    Push this file to a Space with requirements.txt
"""

import gradio as gr
import torch
from transformers import pipeline

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "./bert-agnews-final"   # path saved after training in the notebook
DEVICE     = 0 if torch.cuda.is_available() else -1

LABEL_EMOJI = {
    "World":    "🌍",
    "Sports":   "⚽",
    "Business": "💼",
    "Sci/Tech": "🔬",
}

EXAMPLES = [
    "NASA launches new Mars mission with upgraded AI-powered rover",
    "Stock markets tumble after Federal Reserve raises interest rates sharply",
    "England beats Australia in thrilling Ashes cricket series finale",
    "World leaders meet at UN summit to discuss global climate policy",
    "Apple unveils new iPhone model with on-device AI camera features",
    "Premier League transfers: Manchester City sign star striker for £80m",
    "Tech giants face antitrust scrutiny from EU regulators over data practices",
    "Earthquake measuring 6.5 hits coastal region, tsunami warning issued",
]


def predict(text: str):
    """Return label probabilities for a news headline."""
    if not text.strip():
        return {}
    results = classifier(text)[0]
    return {
        f"{LABEL_EMOJI.get(r['label'], '')} {r['label']}": round(r["score"], 4)
        for r in sorted(results, key=lambda x: -x["score"])
    }


# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from '{MODEL_PATH}' ...")
classifier = pipeline(
    task="text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=DEVICE,
    top_k=None,
)
print("Model loaded ✓")

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    title="BERT News Topic Classifier",
) as demo:

    gr.Markdown(
        """
        # 📰 BERT News Topic Classifier
        Fine-tuned **bert-base-uncased** on the [AG News](https://huggingface.co/datasets/ag_news) dataset.
        Enter any news headline below to classify it into one of four categories.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="News Headline",
                placeholder="e.g. Scientists discover new exoplanet in habitable zone...",
                lines=3,
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Classify", variant="primary")
                clear_btn  = gr.ClearButton([text_input])

        with gr.Column(scale=2):
            label_output = gr.Label(
                num_top_classes=4,
                label="Topic Probabilities",
            )

    gr.Examples(
        examples=[[e] for e in EXAMPLES],
        inputs=text_input,
        label="Try these examples →",
    )

    gr.Markdown(
        """
        ---
        **Categories:** 🌍 World &nbsp;|&nbsp; ⚽ Sports &nbsp;|&nbsp; 💼 Business &nbsp;|&nbsp; 🔬 Sci/Tech  
        **Model:** `bert-base-uncased` fine-tuned for 3 epochs · Accuracy ~94%
        """
    )

    submit_btn.click(fn=predict, inputs=text_input, outputs=label_output)
    text_input.submit(fn=predict, inputs=text_input, outputs=label_output)


if __name__ == "__main__":
    demo.launch(share=False)
