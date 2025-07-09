
# 💳 Credit Card Fraud Detection with Machine Learning

This project is an end-to-end implementation of a machine learning pipeline to detect credit card fraud. Fraudulent transactions are rare and difficult to identify due to the highly imbalanced nature of the dataset, making it a challenging and real-world relevant problem.

---

## 📌 Project Overview

Credit card fraud costs billions annually. Accurately detecting fraudulent transactions is critical to prevent losses and build trust in financial systems. In this project, we use both **supervised** and **unsupervised** machine learning models to identify fraudulent activity in anonymized transaction data.

### 🎯 Objective
To build robust machine learning models that can detect fraudulent transactions with **high recall** while minimizing false positives.

---

## 📊 Dataset

- Sourced from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions (only 492 frauds → ~0.17%)
- Features `V1` to `V28` are PCA-anonymized
- `Amount` is the transaction value, `Class` is the target (1 = fraud, 0 = normal)

---

## 🔍 Key Challenges

- ⚠️ **Class Imbalance**: Very few fraud cases
- 🧠 **Uninterpretable Features**: Anonymized via PCA
- 📉 **Need for High Recall**: False negatives are more dangerous than false positives

---

## 🛠️ ML Techniques Used

- 🧼 Data Cleaning & Preprocessing
- ⚖️ **SMOTE Oversampling** to balance the dataset
- 📈 **Exploratory Data Analysis** with heatmaps, boxplots, and class distributions
- 🤖 Model Building:
  - Logistic Regression (Baseline)
  - Random Forest
  - XGBoost
  - Autoencoder (Unsupervised)
- 🧪 Model Evaluation:
  - Precision, Recall, F1-score
  - ROC Curve Comparison
- 🎛️ Threshold tuning (Autoencoder)
- 📉 Imbalance mitigation strategies

---

## 📊 Model Performance Summary

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.631    | 0.029     | 0.647  | 0.056    |
| Random Forest      | 0.983    | 0.000     | 0.000  | 0.000    |
| Autoencoder        | 0.891    | 0.040     | 0.235  | 0.068    |

> Logistic Regression (with SMOTE) gave the best fraud recall. Random Forest underfit the fraud class. The Autoencoder, trained only on normal data, detected anomalies with modest success.

---

## 📌 Final Conclusion

This project successfully demonstrated how to approach imbalanced classification problems using both traditional supervised models and unsupervised anomaly detection techniques.

- **Logistic Regression with SMOTE** provided the best fraud detection (recall ~65%)
- **Random Forest** underperformed even with resampling — more tuning required
- **Autoencoder** offered an effective unsupervised baseline

> Fraud detection is not about perfect accuracy — it's about **catching as many frauds as possible** while keeping false alarms manageable.

---

## 🚀 Future Work

- ✅ Threshold tuning for Autoencoder to improve precision-recall tradeoff
- ✅ Deploy via Streamlit or Flask for real-time prediction
- ✅ Add feature importance interpretation (SHAP)
- ✅ Explore hybrid models (Autoencoder + Logistic stacking)

---

## 🧠 Author

**Aryan Tiwari**  
*Machine Learning Engineer*  
[GitHub Profile](https://github.com/aryantiwari10)
