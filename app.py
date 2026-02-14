import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.preprocess import preprocess_data
from model.logistic_regression import build_model as lr_model
from model.decision_tree import build_model as dt_model
from model.knn import build_model as knn_model
from model.naive_bayes import build_model as nb_model
from model.random_forest import build_model as rf_model
from model.xgboost_model import build_model as xgb_model


st.title("Bank Marketing Classification App")

st.warning("Upload TEST dataset only (CSV format).")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=';')

    X, y = preprocess_data(df)

    model_choice = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    if model_choice == "Logistic Regression":
        model = lr_model()
    elif model_choice == "Decision Tree":
        model = dt_model()
    elif model_choice == "K-Nearest Neighbors":
        model = knn_model()
    elif model_choice == "Naive Bayes":
        model = nb_model()
    elif model_choice == "Random Forest":
        model = rf_model()
    else:  # model_choice == "XGBoost"
        model = xgb_model()

    # Train model on uploaded dataset
    model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    st.subheader("Evaluation Metrics")

    st.write({
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "AUC": round(roc_auc_score(y, y_prob), 4),
        "Precision": round(precision_score(y, y_pred), 4),
        "Recall": round(recall_score(y, y_pred), 4),
        "F1 Score": round(f1_score(y, y_pred), 4),
        "MCC": round(matthews_corrcoef(y, y_pred), 4)
    })

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
