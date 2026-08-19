# Credit-Card-Fraud-Detection-and-Analysis
End-to-end ML pipeline for detecting credit card fraud in imbalanced data — cleaning, EDA, class balancing (under/over-sampling), and comparison of Logistic Regression, Decision Tree, Random Forest, and KNN.
# Credit Card Fraud Detection and Analysis

An end-to-end machine learning project for detecting fraudulent credit card
transactions in highly imbalanced data.

## Overview

- Cleaned and explored a transaction dataset (~7,000 records, ~0.5% fraud rate)
- Detected outliers (IQR method) and analyzed feature correlation
- Handled severe class imbalance using random under-sampling and over-sampling
- Trained and compared 4 classifiers: Logistic Regression, Decision Tree,
  Random Forest, and KNN
- Evaluated models using Precision, Recall, F1-score, ROC-AUC, and confusion
  matrices

## Results

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------------------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  |
| Decision Tree       | 0.9972   | 0.8333    | 0.5556 | 0.6667   | 0.7775  |
| Random Forest       | 0.9983   | 1.0000    | 0.6667 | 0.8000   | 1.0000  |
| KNN                 | 0.9994   | 1.0000    | 0.8889 | 0.9412   | 0.9444  |

**Best model (by F1-score): Logistic Regression**

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Project Structure

\`\`\`
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb   # full analysis, code + outputs
├── data/
│   └── creditcard.csv                      # transaction dataset
├── reports/
│   └── Credit_Card_Fraud_Detection_Report.docx   # full written report
├── requirements.txt
└── README.md
\`\`\`

## Pipeline

1. **Data Understanding & Cleaning** — missing value handling, duplicate removal
2. **Preprocessing & Feature Engineering** — outlier detection, correlation analysis, scaling, train-test split
3. **Exploratory Data Analysis** — transaction, amount, and time-based fraud patterns
4. **Class Imbalance Handling** — random under-sampling and over-sampling
5. **Modeling** — Logistic Regression, Decision Tree, Random Forest, KNN
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices

## Getting Started

\`\`\`bash
git clone <your-repo-url>
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook notebooks/Credit_Card_Fraud_Detection.ipynb
\`\`\`

## Note on Data

The dataset used here is synthetically generated to match the structure and
class imbalance (~0.5% fraud) of real-world credit card transaction data,
since real card data is sensitive and cannot be redistributed. The pipeline
is designed to work unchanged on a real dataset of the same schema
(`Time`, `V1`-`V28`, `Amount`, `Class`).

## Author

Harshit — B.Tech Computer Science Engineering, RJIT, BSF Academy
