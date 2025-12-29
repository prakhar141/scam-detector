import streamlit as st
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(page_title="📛 Scam Detector", layout="centered")
st.title("📛 Scam Detection System")

REPO_ID = "prakhar146/Scam"        # Hugging Face DATASET repo
REPO_TYPE = "dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = [
    "authority_name",
    "threat_type",
    "time_pressure",
    "payment_method",
    "language_mixing"
]

# --------------------------------------------------
# Load Model, Tokenizer, Calibration
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_all():
    # Tokenizer (uses tokenizer.json, vocab.json, merges.txt)
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID,
        repo_type=REPO_TYPE
    )

    # Model (uses config.json + model.safetensors)
    model = AutoModelForSequenceClassification.from_pretrained(
        REPO_ID,
        repo_type=REPO_TYPE
    ).to(DEVICE)
    model.eval()

    # Calibration file (scam_v1.json)
    calib_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="scam_v1.json"
    )

    with open(calib_path, "r") as f:
        cal = json.load(f)

    temperature = float(cal.get("temperature", 1.0))
    thresholds = np.array(
        cal.get("thresholds", [0.5] * model.config.num_labels)
    )

    return model, tokenizer, temperature, thresholds


model, tokenizer, temperature, base_thresholds = load_all()

# --------------------------------------------------
# Smart Threshold Strategy (Senior-Level)
# --------------------------------------------------
def adaptive_thresholds(probs, base):
    """
    Strategy:
    - If model is uncertain overall → be sensitive
    - If model is confident → be strict
    """
    mean_conf = probs.mean(axis=1, keepdims=True)

    dynamic = np.where(
        mean_conf < 0.15, base - 0.15,      # catch subtle scams
        np.where(mean_conf > 0.45, base + 0.1, base)
    )

    return np.clip(dynamic, 0.2, 0.8)

# --------------------------------------------------
# Prediction Logic
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
        outputs = model(**inputs)
        logits = outputs.logits / temperature
        probs = torch.sigmoid(logits).cpu().numpy()

    thresholds = adaptive_thresholds(probs, base_thresholds)
    preds = (probs > thresholds).astype(int)[0]

    detected_dims = [
        LABEL_NAMES[i] for i, v in enumerate(preds) if v == 1
    ]

    # High-level verdict
    if len(detected_dims) == 0:
        verdict = "🟢 SAFE"
    elif len(detected_dims) <= 2:
        verdict = "🟡 SUSPICIOUS"
    else:
        verdict = "🔴 SCAM"

    return verdict, detected_dims, probs[0]

# --------------------------------------------------
# UI
# --------------------------------------------------
user_text = st.text_area(
    "Paste SMS / WhatsApp / Email text",
    height=150,
    placeholder="⚠️ URGENT: Your bank account will be blocked..."
)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            verdict, dims, probs = predict(user_text)

        st.subheader("Result")
        st.markdown(f"### {verdict}")
        st.write("**Detected Scam Dimensions:**", dims or "None")

        st.write("**Probabilities:**")
        for name, p in zip(LABEL_NAMES, probs):
            st.write(f"- {name}: `{p:.3f}`")
