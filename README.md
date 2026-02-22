# ML Assignment 2 – Bank Marketing Classification

## Live Streamlit App
[(Streamlit App)](https://swsi3kjrlfmmxvvwvu3d3q.streamlit.app/)

---

## 1. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models using the Bank Marketing dataset.

The goal is to predict whether a client will subscribe to a term deposit (binary classification problem).

---

## 2. Dataset Description

The dataset used in this project is the **Bank Marketing Dataset** obtained from the UCI Machine Learning Repository.

### Dataset Characteristics:
- Number of instances: 41,188
- Number of features: 20 input features + 1 target variable
- Target variable: `y`
  - 1 → Client subscribed
  - 0 → Client did not subscribe

The dataset contains customer information and marketing campaign details from a banking institution.

---

## 3. Models Used and Evaluation Metrics

The following classification models were implemented:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## 4. Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.9104 | 0.9317 | 0.6678 | 0.4192 | 0.5151 | 0.4840 |
| Decision Tree | 0.8888 | 0.7227 | 0.5102 | 0.5080 | 0.5091 | 0.4464 |
| KNN | 0.8997 | 0.8569 | 0.5858 | 0.3978 | 0.4738 | 0.4303 |
| Naive Bayes | 0.8505 | 0.8497 | 0.3976 | 0.6149 | 0.4829 | 0.4133 |
| Random Forest | 0.9129 | 0.9432 | 0.6472 | 0.5122 | 0.5719 | 0.5286 |
| XGBoost | 0.9183 | 0.9488 | 0.6688 | 0.5550 | 0.6066 | 0.5645 |

---

## 5. Observations

### Logistic Regression
Performs strongly with high AUC and balanced performance, but recall is relatively lower due to class imbalance.

### Decision Tree
Moderate performance; may suffer from overfitting and lower AUC compared to ensemble methods.

### KNN
Reasonable performance but sensitive to data imbalance and scaling.

### Naive Bayes
Higher recall but lower precision and accuracy compared to other models.

### Random Forest
Improves overall performance through ensemble learning and reduces variance.

### XGBoost
Best performing model across most metrics including Accuracy, AUC, F1 Score, and MCC.

---

## 6. Conclusion

Among all implemented models, **XGBoost** achieved the best overall performance for this dataset, followed closely by **Random Forest**. Ensemble techniques demonstrated superior generalization capability compared to individual classifiers.

---

## Repository Structure
```
ml-assignment-2-bank-marketing/
│-- app.py
|-- bank-additional-full.csv (test dataset)
│-- scaler.pkl
│-- logistic_regression.pkl
│-- decision_tree.pkl
│-- knn.pkl
│-- naive_bayes.pkl
│-- random_forest.pkl
│-- xgboost.pkl
│-- requirements.txt
│-- README.md
│-- model/
  │-- ML_Assignment_2_Bank_Marketing.ipynb
```

---

## How to Run Locally

pip install -r requirements.txt
streamlit run app.py


---



