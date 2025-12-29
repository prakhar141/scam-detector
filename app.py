# streamlit_app.py
import streamlit as st
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="📛 Scam Detector", layout="centered")
st.title("📛 Scam Detection System")

REPO_ID = "prakhar146/Scam"   # Hugging Face DATASET repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = [
    "authority_name",
    "threat_type",
    "time_pressure",
    "payment_method",
    "language_mixing"
]

# --------------------------------------------------
# Load model, tokenizer, calibration
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_all():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID,
        repo_type="dataset"
    )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        REPO_ID,
        repo_type="dataset"
    )
    model.to(DEVICE)
    model.eval()

    # Calibration
    with open(
        torch.hub.download_url_to_file(
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/scam_v1.json",
            "scam_v1.json"
        ),
        "r"
    ) as f:
        cal = json.load(f)

    temperature = float(cal.get("temperature", 1.0))
    thresholds = np.array(cal.get("thresholds", [0.5] * model.config.num_labels))

    return model, tokenizer, temperature, thresholds


model, tokenizer, temperature, base_thresholds = load_all()

# --------------------------------------------------
# Senior ML adaptive thresholding
# --------------------------------------------------
def smart_thresholds(probs, base):
    """
    Strategy:
    - If model is very confident overall → stricter
    - If low confidence but structured signal → looser
    """
    mean_conf = probs.mean(axis=1, keepdims=True)

    # Dynamic scaling
    dynamic = np.where(
        mean_conf > 0.4, base + 0.1,
        np.where(mean_conf < 0.15, base - 0.15, base)
    )

    return np.clip(dynamic, 0.2, 0.8)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits / temperature
        probs = torch.sigmoid(logits).cpu().numpy()

    thresh = smart_thresholds(probs, base_thresholds)
    preds = (probs > thresh).astype(int)

    detected = [LABEL_NAMES[i] for i, v in enumerate(preds[0]) if v == 1]

    # High-level verdict
    if len(detected) == 0:
        verdict = "🟢 SAFE"
    elif len(detected) <= 2:
        verdict = "🟡 SUSPICIOUS"
    else:
        verdict = "🔴 SCAM"

    return verdict, detected, probs[0]


# --------------------------------------------------
# UI
# --------------------------------------------------
user_text = st.text_area(
    "Enter message text",
    height=150,
    placeholder="Paste SMS / WhatsApp / Email text here..."
)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            verdict, detected, probs = predict(user_text)

        st.subheader("Result")
        st.markdown(f"### {verdict}")
        st.write("**Detected Scam Dimensions:**", detected if detected else "None")
        st.write("**Probabilities:**")
        for lbl, p in zip(LABEL_NAMES, probs):
            st.write(f"- {lbl}: {p:.3f}")
