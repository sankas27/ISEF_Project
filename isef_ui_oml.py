# -*- coding: utf-8 -*-
"""ISEF - ui oml

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lG6kfeHicYWdkYMiqhvCv-G_ItSK6qsE
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64
import joblib
import sys
import types
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel

# ✅ Step 1: Define k-mer tokenizer function (MUST MATCH TRAINING)
def kmer_tokenizer(sequence, k=4):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# ✅ Step 2: Custom Joblib Unpickler That Fixes "Can't Get Attribute 'kmer_tokenizer'"
import pickle

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "kmer_tokenizer":
            return kmer_tokenizer  # ✅ Inject `kmer_tokenizer`
        return super().find_class(module, name)

# ✅ Step 3: Define Paths for Model Files
model_path = "./polyreactivity_modeltwo.joblib"
vectorizer_path = "./vectorizer.joblib"
selector_path = "./feature_selector.joblib"

# ✅ Step 4: Load the Vectorizer First (Fixing Missing `kmer_tokenizer`)
try:
    with open(vectorizer_path, "rb") as f:
        vectorizer = CustomUnpickler(f).load()  # ✅ Now correctly passes file handle
    print("✅ Vectorizer loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Vectorizer loading failed: {e}")

# ✅ Step 5: Load the Feature Selector
try:
    with open(selector_path, "rb") as f:
        selector = CustomUnpickler(f).load()  # ✅ Now correctly passes file handle
    print("✅ Feature Selector loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Feature Selector loading failed: {e}")

# ✅ Step 6: Load the Model AFTER the Vectorizer is Fully Loaded
try:
    with open(model_path, "rb") as f:
        model = CustomUnpickler(f).load()  # ✅ Now correctly passes file handle
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Model loading failed: {e}")


# ✅ Function to Transform Sequences Using the Pre-Trained Vectorizer and Feature Selector
def extract_kmer_features(sequence):
    X_kmer = vectorizer.transform([sequence])  # Use the saved vectorizer
    X_selected = selector.transform(X_kmer)   # Select only top 50 features
    return X_selected

# ✅ Function to Compute Confidence Intervals Using Bootstrapping
def compute_confidence_interval(X_kmer, model, bootstrap_samples=1000):
    bootstrap_preds = []
    np.random.seed(42)

    for _ in range(bootstrap_samples):
        sample_indices = np.random.choice(len(X_kmer.toarray()), len(X_kmer.toarray()), replace=True)
        sample_X = X_kmer.toarray()[sample_indices]
        sample_pred = model.predict(sample_X)
        bootstrap_preds.append(sample_pred[0])

    # Compute 95% confidence interval
    lower_bound, upper_bound = np.percentile(bootstrap_preds, [2.5, 97.5])
    return round(lower_bound, 3), round(upper_bound, 3)

# ✅ Function to Generate Feature Importance Visualization
def generate_feature_importance_plot(model, vectorizer):
    feature_importance = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()

    # Get top 10 important features
    sorted_indices = np.argsort(feature_importance)[-10:]
    top_features = [feature_names[i] for i in sorted_indices]
    top_importance = feature_importance[sorted_indices]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_importance, color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("K-mer Features")
    plt.title("Top 10 Influential Sequence Features")

    # Save plot as an image
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    return plot_url

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Define Ranking Thresholds
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.7

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict polyreactivity"""
    try:
        data = request.get_json()
        sequence = data.get("sequence")

        if not sequence:
            return jsonify({"error": "No sequence provided"}), 400

        # ✅ Extract k-mer features using the pre-trained vectorizer
        X_kmer = extract_kmer_features(sequence)

        # ✅ Debug: Check feature shape before prediction
        print(f"Input Features Shape: {X_kmer.shape}")
        print(f"Model Expected Features: {model.n_features_in_}")

        # ✅ Check if feature count matches before predicting
        if X_kmer.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Feature mismatch: Model expects {model.n_features_in_} features, but got {X_kmer.shape[1]}."}), 400

        # ✅ Predict polyreactivity
        predicted_value = model.predict(X_kmer)[0]

        # ✅ Compute confidence interval
        lower_bound, upper_bound = compute_confidence_interval(X_kmer, model)

        # ✅ Assign ranking category
        category = "Low" if upper_bound < LOW_THRESHOLD else "High" if lower_bound > HIGH_THRESHOLD else "Medium"

        # ✅ Compute percentile ranking (assumes you have ranking data)
        percentile_rank = np.random.randint(1, 100)  # Placeholder until real ranking data is available

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
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask Directly (NO Ngrok Needed in Render)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)