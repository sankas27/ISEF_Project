# -*- coding: utf-8 -*-
"""Science Project - Web Interface Development

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13PbnYtFF22eur189E72UeDlGIb7YqUPr
"""

!pip install pyngrok

from google.colab import drive
drive.mount("/content/drive")

!pip install pyngrok
!pip install flask_cors
!ngrok authtoken 2tWGXqWH7UyP0tSkbYCuthu52wy_6BXCjbeRdJcZHBuqZZ4yx

from flask import Flask, request, jsonify
from pyngrok import ngrok
import numpy as np
import pandas as pd
import os
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from sklearn.feature_selection import SelectFromModel

# ✅ Define k-mer tokenizer function (MUST MATCH TRAINING)
def kmer_tokenizer(sequence, k=4):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# ✅ Load trained model, vectorizer, and feature selector
model_path = "/content/drive/MyDrive/polyreactivity_model.joblib"
vectorizer_path = "/content/drive/MyDrive/vectorizer.joblib"
selector_path = "/content/drive/MyDrive/feature_selector.joblib"

if os.path.exists(model_path):
    model = load(model_path)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

if os.path.exists(vectorizer_path):
    vectorizer = load(vectorizer_path)
    print("✅ Vectorizer loaded successfully!")
else:
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

if os.path.exists(selector_path):
    selector = load(selector_path)
    print("✅ Feature Selector loaded successfully!")
else:
    raise FileNotFoundError(f"Feature selector file not found: {selector_path}")

# ✅ Function to Transform Sequences Using the Pre-Trained Vectorizer and Feature Selector
def extract_kmer_features(sequence):
    X_kmer = vectorizer.transform([sequence])  # Use the saved vectorizer
    X_selected = selector.transform(X_kmer)   # Select only top 50 features
    return X_selected

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ API Endpoint for Predictions
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

        return jsonify({
            "sequence": sequence,
            "predicted_value": predicted_value
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Start Flask in a Background Thread
from threading import Thread
import time

def run_flask():
    app.run(port=5000, debug=True, use_reloader=False)

Thread(target=run_flask).start()

# ✅ Start Ngrok after Flask is running
time.sleep(3)
public_url = ngrok.connect(5000).public_url
print(f"✅ Public URL: {public_url}")

from flask import Flask, request, jsonify
from pyngrok import ngrok
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from sklearn.feature_selection import SelectFromModel

# ✅ Define k-mer tokenizer function (MUST MATCH TRAINING)
def kmer_tokenizer(sequence, k=4):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# ✅ Load trained model, vectorizer, and feature selector
model_path = "/content/drive/MyDrive/polyreactivity_model.joblib"
vectorizer_path = "/content/drive/MyDrive/vectorizer.joblib"
selector_path = "/content/drive/MyDrive/feature_selector.joblib"

if os.path.exists(model_path):
    model = load(model_path)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

if os.path.exists(vectorizer_path):
    vectorizer = load(vectorizer_path)
    print("✅ Vectorizer loaded successfully!")
else:
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

if os.path.exists(selector_path):
    selector = load(selector_path)
    print("✅ Feature Selector loaded successfully!")
else:
    raise FileNotFoundError(f"Feature selector file not found: {selector_path}")

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

# ✅ Start Flask in a Background Thread
from threading import Thread
import time

def run_flask():
    app.run(port=5000, debug=True, use_reloader=False)

Thread(target=run_flask).start()

# ✅ Start Ngrok after Flask is running
time.sleep(3)
public_url = ngrok.connect(5000).public_url
print(f"✅ Public URL: {public_url}")

!fuser -k 5000/tcp

!pkill -f ngrok

from joblib import load

model_path = "/content/drive/MyDrive/polyreactivity_model.joblib"
model = load(model_path)
print("✅ Model loaded successfully!")

print("🔍 Model Expected Features:", model.n_features_in_)

!curl -X POST "https://57c3-34-42-161-91.ngrok-free.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYAFTSYWMH"}'

print("✅ Checking if model is trained...")
print("Model Type:", type(model))
print("Number of Trees:", getattr(model, "n_estimators", "Not Available"))
print("Estimators Attribute Exists:", hasattr(model, "estimators_"))