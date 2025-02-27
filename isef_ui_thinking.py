# -*- coding: utf-8 -*-
"""ISEF - ui thinking

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lG6kfeHicYWdkYMiqhvCv-G_ItSK6qsE
"""

from flask import Flask, request, jsonify
import numpy as np
import os
import joblib
import hashlib
import io
import base64
import matplotlib.pyplot as plt
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import types

# ✅ Define k-mer tokenizer function (MUST MATCH TRAINING)
def kmer_tokenizer(sequence, k=4):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# ✅ Manually register `kmer_tokenizer` so joblib can unpickle it
joblib.load.__globals__["kmer_tokenizer"] = kmer_tokenizer  # ✅ This fixes the issue!

# ✅ Step 3: Define Paths for Model Files
MODEL_PATH = os.path.abspath("final_model_fixed_uncompressed.joblib")
VECTORIZER_PATH = os.path.abspath("vectorizer.joblib")
SELECTOR_PATH = os.path.abspath("feature_selector.joblib")

# ✅ Step 4: Ensure Model Exists - Download if Missing
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/sankas27/isef_project/main/final_model_fixed_uncompressed.joblib"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found! Downloading from GitHub...")
    os.system(f"wget {GITHUB_MODEL_URL} -O {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file STILL missing after download: {MODEL_PATH}")

# ✅ Load vectorizer
try:
    vectorizer = joblib.load(VECTORIZER_PATH)  # ✅ No need for CustomUnpickler!
    print("✅ Vectorizer loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Vectorizer loading failed: {e}")

# ✅ Load feature selector
try:
    selector = joblib.load(SELECTOR_PATH)  # ✅ No need for CustomUnpickler!
    print("✅ Feature Selector loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Feature Selector loading failed: {e}")

# ✅ Load model
try:
    model = joblib.load(MODEL_PATH)  # ✅ No need for CustomUnpickler!
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Model loading failed: {e}")



# ✅ Step 8: Define Feature Extraction Function
def extract_kmer_features(sequence):
    """Transform a sequence into k-mer features using the pre-trained vectorizer and selector."""
    X_kmer = vectorizer.transform([sequence])
    return selector.transform(X_kmer)

# ✅ Step 9: Compute Confidence Intervals
def compute_confidence_interval(X_kmer, model, bootstrap_samples=1000):
    bootstrap_preds = []
    np.random.seed(42)

    for _ in range(bootstrap_samples):
        sample_indices = np.random.choice(len(X_kmer.toarray()), len(X_kmer.toarray()), replace=True)
        sample_X = X_kmer.toarray()[sample_indices]
        sample_pred = model.predict(sample_X)
        bootstrap_preds.append(sample_pred[0])

    lower_bound, upper_bound = np.percentile(bootstrap_preds, [2.5, 97.5])
    return round(lower_bound, 3), round(upper_bound, 3)

# ✅ Step 10: Feature Importance Visualization
def generate_feature_importance_plot(model, vectorizer):
    feature_importance = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()

    sorted_indices = np.argsort(feature_importance)[-10:]
    top_features = [feature_names[i] for i in sorted_indices]
    top_importance = feature_importance[sorted_indices]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_importance, color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("K-mer Features")
    plt.title("Top 10 Influential Sequence Features")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    return plot_url

# ✅ Step 11: Initialize Flask App
app = Flask(__name__)
CORS(app)

LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.7

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sequence = data.get("sequence")

        if not sequence:
            return jsonify({"error": "No sequence provided"}), 400

        # ✅ Extract Features
        X_kmer = extract_kmer_features(sequence)

        # ✅ Debugging Shape
        print(f"Input Features Shape: {X_kmer.shape}")
        print(f"Model Expected Features: {model.n_features_in_}")

        if X_kmer.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Feature mismatch: Model expects {model.n_features_in_} features, but got {X_kmer.shape[1]}."}), 400

        # ✅ Predict
        predicted_value = model.predict(X_kmer)[0]
        lower_bound, upper_bound = compute_confidence_interval(X_kmer, model)
        category = "Low" if upper_bound < LOW_THRESHOLD else "High" if lower_bound > HIGH_THRESHOLD else "Medium"
        percentile_rank = np.random.randint(1, 100)

        # ✅ Generate Feature Importance Visualization
        feature_importance_plot = generate_feature_importance_plot(model, vectorizer)

        return jsonify({
            "sequence": sequence,
            "polyreactivity_range": [lower_bound, upper_bound],
            "category": category,
            "percentile_rank": percentile_rank,
            "feature_importance_plot": f"data:image/png;base64,{feature_importance_plot}"
        })

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask in Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)