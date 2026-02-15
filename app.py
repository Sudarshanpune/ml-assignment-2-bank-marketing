import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Bank Marketing Classification App")

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=";")

    # Convert target
    data['y'] = data['y'].map({'yes': 1, 'no': 0})

    # Encode categorical
    from sklearn.preprocessing import LabelEncoder
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    X = data.drop('y', axis=1)
    y = data['y']

    # Load scaler
    scaler = joblib.load("scaler.pkl")
    X_scaled = scaler.transform(X)

    # Load model
    model_dict = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    model = joblib.load(model_dict[model_option])

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", acc)
    st.write("AUC:", auc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)
    st.write("MCC:", mcc)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    st.pyplot(fig)
