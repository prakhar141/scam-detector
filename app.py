# streamlit_app.py
import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.set_page_config(page_title="📛 Scam Detector", layout="wide")
st.title("📛 Scam Detection & Classification")

# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_calibration(model_path, calibration_path, tokenizer_path):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Load calibration
    with open(calibration_path, "r") as f:
        cal = json.load(f)
    
    thresholds = np.array(cal.get("thresholds", [0.5] * model.config.num_labels))
    temperature = float(cal.get("temperature", 1.0))
    
    return model, tokenizer, thresholds, temperature

MODEL_PATH     = "https://huggingface.co/datasets/prakhar146/Scam/resolve/main/model.safetensors"
TOKENIZER_PATH = "https://huggingface.co/datasets/prakhar146/Scam/resolve/main/tokenizer"
CALIB_PATH     = "https://huggingface.co/datasets/prakhar146/Scam/resolve/main/scam_v1.json"

model, tokenizer, thresholds, temperature = load_model_and_calibration(MODEL_PATH, CALIB_PATH, TOKENIZER_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# Smart Threshold Strategy
# -----------------------------
def adaptive_thresholds(probs, base_thresh=0.5):
    """
    Adjust thresholds to highlight rare suspicious patterns.
    - For very low probabilities but non-zero, amplify signal.
    - For common low probs, stay conservative.
    """
    # Example: scale threshold by percentile
    perc_75 = np.percentile(probs, 75)
    adaptive_thresh = np.minimum(base_thresh, perc_75)
    return adaptive_thresh

# -----------------------------
# Prediction Function
# -----------------------------
def predict(texts):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"] / temperature
        probs = torch.sigmoid(logits).cpu().numpy()

    # Apply adaptive thresholding
    adaptive_thresh = adaptive_thresholds(probs, thresholds)
    preds = (probs > adaptive_thresh).astype(int)
    
    # Map labels
    label_names = ["authority_name", "threat_type", "time_pressure", "payment_method", "language_mixing"]
    results = []
    for i, text in enumerate(texts):
        detected = [label_names[j] for j, val in enumerate(preds[i]) if val == 1]
        # High-level classification
        if len(detected) == 0:
            category = "Safe"
        elif len(detected) < 3:
            category = "Suspicious"
        else:
            category = "Scam"
        
        results.append({
            "text": text,
            "category": category,
            "detected_dimensions": detected,
            "probabilities": probs[i].tolist()
        })
    return results

# -----------------------------
# Streamlit UI
# -----------------------------
st.subheader("Enter text for scam detection:")
user_input = st.text_area("Type or paste text here", "", height=150)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing…"):
            output = predict([user_input])
            for res in output:
                st.write("**Input:**", res["text"])
                st.write("**Category:**", res["category"])
                st.write("**Detected Dimensions:**", res["detected_dimensions"])
                st.write("**Probabilities:**", np.round(res["probabilities"], 4))
                st.markdown("---")
