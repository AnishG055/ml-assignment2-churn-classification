import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Logistic Regression": ROOT / "model" / "logistic_regression.joblib",
    "Decision Tree": ROOT / "model" / "decision_tree.joblib",
    "kNN": ROOT / "model" / "knn.joblib",
    "Naive Bayes": ROOT / "model" / "naive_bayes.joblib",
    "Random Forest": ROOT / "model" / "random_forest.joblib",
    "SVM (provisional)": ROOT / "model" / "svm_provisional.joblib",
}

FEATURE_COLUMNS = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

st.set_page_config(page_title="ML Assignment 2", page_icon="📊", layout="wide")
st.title("Machine Learning Classification Dashboard")
st.caption("UCI Breast Cancer Wisconsin (Diagnostic) dataset")

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Select model", list(MODEL_PATHS))
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

st.write(
    "Upload the test CSV generated for the assignment. The file should contain "
    "the 30 feature columns and a diagnosis column with M (malignant) or B (benign)."
)

if uploaded is None:
    st.info("Upload test_data.csv to evaluate the selected model.")
    st.stop()

df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()
if "diagnosis" not in df.columns:
    st.error("The uploaded CSV must contain a 'diagnosis' column.")
    st.stop()

X = df[FEATURE_COLUMNS]
y_true = df["diagnosis"].astype(str).str.upper().map({"M": 1, "B": 0})
if y_true.isna().any():
    st.error("Diagnosis values must be M or B.")
    st.stop()

model = joblib.load(MODEL_PATHS[model_name])
y_pred = model.predict(X)
y_score = (
    model.predict_proba(X)[:, 1]
    if hasattr(model, "predict_proba")
    else model.decision_function(X)
)

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_score),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1": f1_score(y_true, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
cm_df = pd.DataFrame(cm, index=["Actual B", "Actual M"], columns=["Predicted B", "Predicted M"])
st.dataframe(cm_df, use_container_width=True)

st.subheader("Classification Report")
report = classification_report(
    y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.subheader("Predictions")
out = df.copy()
out["predicted_diagnosis"] = np.where(y_pred == 1, "M", "B")
out["malignant_probability"] = y_score
st.dataframe(out, use_container_width=True)
