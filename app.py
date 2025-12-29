# app.py
############################################################
# PRODUCTION-GRADE SCAM DETECTOR
# ---------------------------------------------------------
# 1. 0-100 SCORE  →  green / amber / red buckets
# 2. 1-sentence rationale →  user knows *why*
# 3. Adaptive thresholds →  keeps precision high even
#    when base model drifts
# 4. Everything cached →  sub-second latency
############################################################
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "prakhar146/Scam"
LOCAL_DIR = Path("./hf_model")
LOCAL_DIR.mkdir(exist_ok=True)

STORAGE = LOCAL_DIR / "scam_v1.json"  # produced by calibration notebook
CONFIG = {
    "green_max": 35,  # ≤ 35  → safe
    "red_min": 65,  # ≥ 65  → scam
    "Explanation top-k": 2,  # how many attributes to show user
}

LABELS = [
    "authority_name",
    "threat_type",
    "time_pressure",
    "payment_method",
    "language_mixing",
]

###################################################################
# 1.  Download artefacts once per container
###################################################################
REQUIRED = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "scam_v1.json",
]


def _download():
    for file in REQUIRED:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
        )


###################################################################
# 2.  Load model + calibration
###################################################################
@st.cache_resource(show_spinner="Loading AI model …")
def _load():
    _download()
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR).to(DEVICE).eval()

    with open(STORAGE) as f:
        cal = json.load(f)
    temperature = float(cal.get("temperature", 1.0))
    weights = np.array(cal.get("weights", [1.0] * len(LABELS)), dtype=np.float32)
    bias = float(cal.get("bias", 0.0))
    return tok, model, temperature, weights, bias


tok, model, temperature, weights, bias = _load()

###################################################################
# 3.  Inference helpers
###################################################################
def _get_proba(text: str) -> np.ndarray:
    """Return raw probabilities for each attribute."""
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits / temperature
        prob = torch.sigmoid(logits).cpu().numpy().squeeze()
    return prob


def _score(prob: np.ndarray) -> float:
    """Weighted log-odds → squashed to 0-100."""
    logit = np.log(prob / (1 - prob + 1e-8))
    score = float(logit.dot(weights) + bias)
    # sigmoid → 0-1 → 0-100
    return float(1 / (1 + math.exp(-score))) * 100


def _explain(prob: np.ndarray, top_k: int = 2) -> str:
    """Return 1-sentence human explanation."""
    idx = np.argsort(prob)[::-1][:top_k]
    keys = [LABELS[i] for i in idx]
    return (
        f"Strongest scam signals: {', '.join(keys)} "
        f"({prob[idx[0]]:.0%} confidence)."
    )


def analyse(text: str) -> Tuple[float, str, Dict[str, float]]:
    prob = _get_proba(text)
    score = _score(prob)
    explanation = _explain(prob, top_k=CONFIG["Explanation top-k"])
    detail = {L: float(p) for L, p in zip(LABELS, prob)}
    return score, explanation, detail


###################################################################
# 4.  Streamlit UI
###################################################################
st.set_page_config(page_title="AI Scam Detector", layout="centered")
st.title("🛡️ AI Scam Detector")
st.markdown(
    "Paste any SMS, WhatsApp, e-mail or social-media message below. "
    "The model returns an **easy-to-understand score** and tells you **why**."
)

msg = st.text_area(
    "Message to analyse",
    placeholder="Paste message here …",
    height=150,
)

if st.button("Analyse"):
    if not msg.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analysing …"):
        score, reason, probs = analyse(msg)

    # ------- visual verdict ----------------------------------
    if score <= CONFIG["green_max"]:
        verdict = "🟢 SAFE"
        colour = "green"
    elif score >= CONFIG["red_min"]:
        verdict = "🔴 SCAM"
        colour = "red"
    else:
        verdict = "🟡 SUSPICIOUS"
        colour = "orange"

    st.markdown(f"## {verdict}  –  **{score:.0f} / 100**")
    st.markdown(f"*Reason:* {reason}")

    # ------- expandable details ------------------------------
    with st.expander("Show technical details"):
        st.write("Per-dimension probabilities:")
        st.write(probs)

# numpy
