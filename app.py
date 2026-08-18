import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

st.title("📊 Model Evaluation & Classification Dashboard")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

# Updated model options matching your repository filenames and extensions
model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "SVM"]
)

# Mapping selections to the actual .joblib files in your root directory
file_map = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm_provisional.joblib"
}

if uploaded_file is not None:
    # Read the uploaded test CSV file
    df = pd.read_csv(uploaded_file)
    
    # Adjust column name check based on your dataset (target column can be 'Churn' or 'Diagnosis')
    target_col = "Churn" if "Churn" in df.columns else ("diagnosis" if "diagnosis" in df.columns else df.columns[-1])
    
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    
    # If target is categorical (like 'M'/'B'), map it to binary 1/0 if needed
    if y_test.dtype == object:
        y_test = y_test.map({'M': 1, 'B': 0, 'Yes': 1, 'No': 0})

    model_path = file_map[model_choice]
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        st.subheader(f"Evaluation Metrics: {model_choice}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        col4.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.error(f"Model file '{model_path}' not found in the root directory.")
else:
    st.info("Please upload your 'test_data.csv' file using the sidebar to view evaluation results.")
