# Machine Learning Assignment 2

## a. Problem statement

Build and evaluate multiple classification models on one public classification dataset and provide an interactive Streamlit application that accepts test data, lets the user select a model, displays evaluation metrics, and shows a confusion matrix/classification report.

> **Important assignment ambiguity:** Section 2 states that 6 models are required, but only 5 models are explicitly named: Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random Forest. The sixth model below is therefore marked **provisional: SVM**. Confirm the intended sixth model with the faculty before final submission.

## b. Dataset description

Dataset: **Breast Cancer Wisconsin (Diagnostic)** from the UCI Machine Learning Repository.

The UCI repository reports **569 instances and 30 real-valued features** for this binary classification dataset. The diagnosis target is M (malignant) or B (benign). Dataset DOI: 10.24432/C5DW2B.

The assignment requires at least 500 instances and 12 features; this dataset meets both thresholds.

Positive class used for metric calculation: **M = malignant (1)**; negative class: **B = benign (0)**.

Reference:
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## c. Github Repository Link

**To be replaced after creating the repository:**
`(https://github.com/AnishG055/ml-assignment2-churn-classification)`

Repository should contain:
- `app.py`
- `train_models.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- `dataset.csv`
- `metrics.csv`
- `model/*.joblib`

## Live Streamlit App Link

**To be replaced after deployment:**
`https://<your-app-name>.streamlit.app/`

## d. Models used

Experimental setup: stratified 80/20 train-test split with `random_state=42`. Standardization is applied through pipelines for Logistic Regression and kNN; an RBF SVM is also standardized.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | **0.9737** | 0.9944 | **1.0000** | 0.9286 | **0.9630** | **0.9442** |
| SVM (provisional) | **0.9737** | 0.9947 | **1.0000** | 0.9286 | **0.9630** | **0.9442** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performs strongly, with 96.49% accuracy and 0.9960 AUC. Scaling helps the linear model operate on features with different numeric ranges. |
| Decision Tree | Has the lowest scores among the compared models. Its accuracy and AUC are still good, but its single-tree structure is less robust on this split than the ensemble methods. |
| kNN | Performs well after standardization. Its 95.61% accuracy and 0.9823 AUC indicate effective local-neighbour classification, although recall is slightly below Logistic Regression and Random Forest. |
| Naive Bayes | Produces perfect precision on this test split but lower recall (0.8333), meaning it misses more positive malignant cases than the top models. |
| Random Forest (Ensemble) | Gives the best overall balance among the five mandated models: 97.37% accuracy, 0.9944 AUC, 1.0000 precision, 0.9286 recall, 0.9630 F1 and 0.9442 MCC. |
| SVM (provisional) | Matches Random Forest on accuracy, precision, recall, F1 and MCC on this split, with slightly higher AUC (0.9947). It is included only to resolve the assignment's unexplained sixth-model count. |
| Overall Winner | **Random Forest** among the five explicitly mandated models. SVM ties it on this particular split, but SVM is provisional and should not replace a faculty-specified sixth model. |

## Streamlit features implemented

1. CSV test-data upload.
2. Model selection dropdown.
3. Accuracy, AUC, precision, recall, F1 and MCC display.
4. Confusion matrix.
5. Classification report.
6. Per-row predictions and malignant-class probability.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Upload `test_data.csv` in the app.

## Deployment

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select the repository and branch.
4. Set the entry point to `app.py`.
5. Deploy.
6. Test the live URL in a private/incognito browser before submitting.

## BITS Virtual Lab screenshot

The assignment specifically requires one screenshot showing execution on BITS Virtual Lab. **This must be captured by the student and cannot be truthfully fabricated here.**

## Academic-integrity note

The assignment explicitly says AI tools are permitted only for learning support and warns against direct copy-paste submissions. The code and explanations should therefore be reviewed, understood, customized, and actually executed by the student before submission. Maintain genuine Git commit history and do not claim a deployment or screenshot that has not actually been completed.
