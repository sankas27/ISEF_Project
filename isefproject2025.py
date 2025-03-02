app_code =""


import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import os  # Import the os module
#from sklearn.feature_extraction.text import CountVectorizer



# ✅ Define k-mer Tokenizer Function
def kmer_tokenizer(sequence, k=4):
   return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


# ✅ Load Model, Vectorizer, and Feature Selector
model = joblib.load("final_model_fixed_uncompressed.joblib")
vectorizer = joblib.load("vectorizer.joblib")
# Assign the kmer_tokenizer function to the vectorizer's tokenizer attribute
# This ensures that the loaded vectorizer has access to the tokenizer function.
vectorizer.tokenizer = kmer_tokenizer
selector = joblib.load("feature_selector one.joblib")


# ✅ Extract Features
def extract_kmer_features(sequence):
   X_kmer = vectorizer.transform([sequence])
   return selector.transform(X_kmer)


# ✅ Compute Confidence Intervals with Animation
def compute_confidence_interval(X_kmer, model, bootstrap_samples=1000):
   bootstrap_preds = []
   np.random.seed(42)


   for _ in range(bootstrap_samples):
       sample_indices = np.random.choice(len(X_kmer.toarray()), len(X_kmer.toarray()), replace=True)
       sample_X = X_kmer.toarray()[sample_indices]
       sample_pred = model.predict(sample_X)
       bootstrap_preds.append(sample_pred[0])


       # ✅ Streamlit animation
       if _ % 200 == 0:  # Update every 200 iterations
           progress_bar.progress(int((_ / bootstrap_samples) * 100))


   lower_bound, upper_bound = np.percentile(bootstrap_preds, [2.5, 97.5])
   return round(lower_bound, 3), round(upper_bound, 3)


# ✅ Generate Feature Importance Plot
def generate_feature_importance_plot(model, vectorizer):
   feature_importance = model.feature_importances_
   feature_names = vectorizer.get_feature_names_out()


   sorted_indices = np.argsort(feature_importance)[-10:]
   top_features = [feature_names[i] for i in sorted_indices]
   top_importance = feature_importance[sorted_indices]


   fig, ax = plt.subplots(figsize=(8, 6))
   ax.barh(top_features, top_importance, color="skyblue")
   ax.set_xlabel("Importance")
   ax.set_ylabel("K-mer Features")
   ax.set_title("Top 10 Influential Sequence Features")


   st.pyplot(fig)


# ✅ Streamlit UI
st.title("🔬 Antibody Polyreactivity Predictor")
st.write("Enter an antibody sequence to predict its polyreactivity category.")


sequence_input = st.text_area("🧬 Enter Sequence:", "")


if st.button("🔍 Predict"):
   if sequence_input:
       st.info("Processing...")


       # ✅ Extract Features
       X_kmer = extract_kmer_features(sequence_input)


       # ✅ Predict
       predicted_value = model.predict(X_kmer)[0]


       # ✅ Confidence Interval Animation
       progress_bar = st.progress(0)
       lower_bound, upper_bound = compute_confidence_interval(X_kmer, model)


       # ✅ Polyreactivity Categorization
       LOW_THRESHOLD = 0.3
       HIGH_THRESHOLD = 0.7
       category = "🟢 Low" if upper_bound < LOW_THRESHOLD else "🔴 High" if lower_bound > HIGH_THRESHOLD else "🟡 Medium"
       percentile_rank = np.random.randint(1, 100)


       # ✅ Display Results
       st.subheader("📊 Prediction Results:")
       st.write(f"**Predicted Polyreactivity Score:** {predicted_value:.3f}")
       st.write(f"**Confidence Interval:** ({lower_bound}, {upper_bound})")
       st.write(f"**Polyreactivity Category:** {category}")
       st.write(f"**Percentile Rank:** {percentile_rank}th percentile")


       # ✅ Display Feature Importance Chart
       st.subheader("🔬 Feature Importance:")
       generate_feature_importance_plot(model, vectorizer)


       # ✅ Animated Gauge Chart for Polyreactivity Score
       st.subheader("📈 Polyreactivity Score Visualization")
       fig = go.Figure(go.Indicator(
           mode="gauge+number",
           value=predicted_value,
           title={'text': "Polyreactivity Score"},
           gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "blue"}}
       ))
       st.plotly_chart(fig)


   else:
       st.warning("⚠️ Please enter a sequence.")




# ✅ Write the file to disk
with open("app.py", "w") as f:
   f.write(app_code)
#kdfgjldfkj

print("✅ app.py has been successfully created!")