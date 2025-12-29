import streamlit as st
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from pathlib import Path

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(page_title="📛 Indian Scam Detector", layout="centered")
st.title("📛 Indian Scam Detection System")
st.caption("Built for real SMS, WhatsApp & Email scams seen in India")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ID = "prakhar146/Scam"
REPO_TYPE = "dataset"

LABELS = [
    "authority_name",   # Police, Bank, Govt, UIDAI
    "threat_type",      # Arrest, block, disconnect
    "time_pressure",    # Today, 2 hours, immediately
    "payment_method",   # UPI, gift cards, crypto
    "language_mixing"   # Hindi + English
]

LOCAL_DIR = Path("./hf_model")
LOCAL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Risk weights (Indian reality aware)
# --------------------------------------------------
RISK_WEIGHTS = {
    "authority_name": 2.5,
    "threat_type": 3.0,
    "time_pressure": 2.0,
    "payment_method": 4.0,   # strongest indicator
    "language_mixing": 0.8   # weak alone
}

# --------------------------------------------------
# Download files
# --------------------------------------------------
REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "scam_v1.json"
]

def download_files():
    for f in REQUIRED_FILES:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f,
            repo_type=REPO_TYPE,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_all():
    download_files()

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
    model.to(DEVICE)
    model.eval()

    with open(LOCAL_DIR / "scam_v1.json", "r") as f:
        cal = json.load(f)

    temperature = float(cal.get("temperature", 1.0))
    return model, tokenizer, temperature

model, tokenizer, temperature = load_all()

# --------------------------------------------------
# Risk computation (senior ML logic)
# --------------------------------------------------
def compute_risk(probs):
    """
    Non-linear, weighted risk aggregation
    """
    total_risk = 0.0
    contributions = {}

    for label, p in zip(LABELS, probs):
        weight = RISK_WEIGHTS[label]

        # Non-linear amplification for high confidence
        signal_strength = (p ** 1.6) * weight
        contributions[label] = signal_strength
        total_risk += signal_strength

    # Strong real-world interaction rules
    if probs[LABELS.index("payment_method")] > 0.6 and \
       probs[LABELS.index("authority_name")] > 0.5:
        total_risk += 3.0  # Police + money = scam

    if probs[LABELS.index("time_pressure")] > 0.6 and \
       probs[LABELS.index("payment_method")] > 0.5:
        total_risk += 2.0  # Urgency + payment

    return total_risk, contributions

# --------------------------------------------------
# Verdict logic (human-aligned)
# --------------------------------------------------
def verdict_from_risk(risk):
    if risk < 2.5:
        return "🟢 SAFE", "Low risk. Message matches normal Indian communication patterns."
    elif risk < 6.0:
        return "🟡 SUSPICIOUS", (
            "Some scam-like patterns detected. "
            "Do NOT click links or share OTPs. Verify independently."
        )
    else:
        return "🔴 SCAM", (
            "High confidence scam. This matches common Indian fraud tactics. "
            "Do NOT respond, pay, or share any information."
        )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def analyze_message(text):
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
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    risk, contributions = compute_risk(probs)
    verdict, explanation = verdict_from_risk(risk)

    detected = [
        lbl for lbl, p in zip(LABELS, probs) if p > 0.5
    ]

    return verdict, explanation, risk, detected, probs, contributions

# --------------------------------------------------
# UI
# --------------------------------------------------
user_text = st.text_area(
    "Paste SMS / WhatsApp / Email message",
    height=160,
    placeholder="Example: Your electricity will be disconnected today. Pay via UPI..."
)

if st.button("Analyze Message"):
    if not user_text.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing message like a fraud investigator..."):
            verdict, explanation, risk, detected, probs, contrib = analyze_message(user_text)

        st.subheader("Final Verdict")
        st.markdown(f"## {verdict}")
        st.write(explanation)

        st.divider()

        st.subheader("Why this result?")
        if detected:
            for lbl in detected:
                st.write(f"• **{lbl.replace('_',' ').title()}** detected")
        else:
            st.write("No strong scam indicators found.")

        st.divider()

        st.subheader("Risk Breakdown")
        for lbl in LABELS:
            st.write(
                f"- {lbl}: probability={probs[LABELS.index(lbl)]:.2f}, "
                f"risk contribution={contrib[lbl]:.2f}"
            )

        st.caption(f"Total Risk Score: {risk:.2f}")
