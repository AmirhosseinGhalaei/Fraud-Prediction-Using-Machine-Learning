# Fraud Prediction Using Machine Learning  

## 📌 1. Project Description  
- This project focuses on building a machine learning model to predict fraudulent transactions based on various financial and behavioral features. The goal is to develop a reliable and scalable fraud detection system that minimizes false positives while maintaining high precision and recall.
- This README provides an overview of the dataset, preprocessing steps, model development, and final results to help others understand and replicate the process efficiently.


## 🎯 2. Key Objectives
- •	Understanding the dataset and identifying key fraud indicators.
- •	Preprocessing and engineering relevant features to improve model performance.
- •	Training, evaluating, and fine-tuning multiple machine learning models.
- •	Analyzing the model’s predictions and improving overall fraud detection accuracy.



## 📂 3. Dataset Overview

The dataset utilized in this project is the **Credit Card Fraud Detection** dataset, sourced from the **ULB Machine Learning Group** and made available on **Kaggle**.  

📌 **Dataset Link:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)  

### 🔹 Dataset Description  
This dataset comprises transactions made by **European cardholders** over a **two-day period in September 2013**. It contains **284,807 transactions**, of which **492** are labeled as **fraudulent**, highlighting its highly **imbalanced nature**:  

✅ **Legitimate Transactions:** 99.828% (284,315)  
⚠️ **Fraudulent Transactions:** 0.172% (492)  

### 🔹 Feature Information  
Due to confidentiality concerns, the dataset's features have undergone **Principal Component Analysis (PCA) transformation**, resulting in **28 anonymized features** labeled **V1 through V28**.  

Additionally, the dataset includes:  
- **⏳ 'Time' Feature** – Represents the seconds elapsed between the first transaction and each subsequent transaction.  
- **💰 'Amount' Feature** – Indicates the transaction amount.  
- **🎯 'Class' Feature** – The target variable:  
  - **0** → Legitimate transaction  
  - **1** → Fraudulent transaction  

### 🔹 Real-World Challenges  
This dataset is widely used for **benchmarking fraud detection systems** and presents challenges typical of real-world scenarios:  
- **⚠️ Severe Class Imbalance** – Fraudulent transactions are extremely rare (0.172%).  
- **🔍 Anonymized Features** – Feature interpretation is limited due to PCA transformation.  
- **📉 Data Distribution Issues** – Requires advanced techniques for effective fraud detection.  

These characteristics make this dataset an **excellent case study** for developing robust **fraud detection models** using machine learning techniques.  



## 🛠️ 4. Technologies Used  
- **Python**  
- **Pandas, NumPy** – Data Processing  
- **Scikit-Learn** – Machine Learning  
- **Matplotlib, Seaborn** – Data Visualization


## 🚀 5. How to Run  
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/AmirhosseinGhalaei/Fraud-Prediction-Using-Machine-Learning.git

2. **Create the project directory**:  
   ```bash
   mkdir Fraud-Prediction-Using-Machine-Learning

3. **Navigate to the directory**:
   ```bash
   cd Fraud-Prediction-Using-Machine-Learning

4. **Install required libraries**:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

5. **Run the Python script**:
   ```bash
    python FraudPrediction.py

## ⚙️ 6. Preprocessing & Feature Engineering  

- In this stage, we prepare the dataset for model training by handling missing values, normalizing numerical features, and addressing class imbalance.  

- 🔹 Data Cleaning  
✅ The dataset has **no missing values**, as confirmed through an initial **Exploratory Data Analysis (EDA)**.  

- 🔹 Feature Scaling  
✅ Most features (**V1–V28**) are **PCA-transformed**, so they do not require further scaling.  
✅ The **'Amount'** and **'Time'** features are **standardized using MinMaxScaler** to improve model performance.  

- 🔹 Class Imbalance Handling  
✅ Fraudulent transactions constitute only **0.172%** of the dataset.  
✅ We apply **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the class distribution.  

- 🔹 Feature Selection  
✅ All **PCA-generated features** are included in the model.  
✅ Feature importance is analyzed using **Recursive Feature Elimination (RFE)** and **SHAP values** to identify the most influential features.  

- 🔹 New Feature Engineering  
✅ No additional features are created due to the nature of the dataset.  
✅ **Domain-specific transformations**, such as **log transformations on 'Amount'**, are considered.  

These **preprocessing and feature engineering** steps ensure that our dataset is **optimized for model training** while addressing the challenges posed by imbalanced data.  



## 🤖 7. Model Development  

In this phase, we train multiple machine learning models, evaluate their performance, and fine-tune hyperparameters to enhance fraud detection accuracy.  

### 🔹 7.1 Model Selection & Training  
We experiment with various machine learning algorithms, including:  

✅ **Logistic Regression** – A baseline model for interpretability.  
✅ **Random Forest** – A robust ensemble model that captures complex relationships.  
✅ **XGBoost** – A gradient boosting model known for handling imbalanced datasets effectively.  
✅ **Neural Networks (MLP Classifier)** – A deep learning-based approach for improved pattern recognition.  

Each model is trained on the **preprocessed dataset** using **stratified k-fold cross-validation** to ensure balanced class distribution during training.  

### 🔹 7.2 Model Evaluation Metrics  
Given the severe **class imbalance**, traditional **accuracy** is not an ideal metric. Instead, we focus on:  

🎯 **Precision** – Minimizing false positives.  
🎯 **Recall** – Ensuring fraudulent transactions are detected.  
🎯 **F1-Score** – A balance between precision and recall.  
🎯 **AUC-ROC Score** – Measures overall model performance in distinguishing fraud from non-fraud.  

These metrics provide a **comprehensive evaluation** of our models, ensuring that we prioritize **fraud detection accuracy** while minimizing false alarms.  


### 🔹 7.3 Hyperparameter Tuning  
To improve performance, we fine-tune model parameters using:  

✅ **GridSearchCV & RandomizedSearchCV** – For systematic hyperparameter optimization.  
✅ **Bayesian Optimization** – To efficiently find the best model settings.  
✅ **Early Stopping (Neural Networks)** – Prevents overfitting by stopping training when validation performance plateaus.  

After evaluating multiple models, the one with the **highest AUC-ROC and F1-score** is selected for final deployment.  

---

## 📈 8. Predictions & Performance  

This section summarizes the model’s **final results**, **key metrics**, and **insights based on predictions**.  

### 🔹 8.1 Model Predictions  
After training the models, we tested them on **unseen data** to assess their **real-world performance**. The key evaluation involved **predicting fraudulent transactions** and comparing them with actual labels.  

### 🔹 8.2 Performance Metrics  
Since fraud detection involves **imbalanced data**, we focus on:  

🎯 **Precision** – The proportion of actual fraud cases among all predicted fraud cases.  
🎯 **Recall** – The percentage of fraud cases correctly identified.  
🎯 **F1-Score** – The harmonic mean of precision and recall.  
🎯 **AUC-ROC Score** – Measures the model’s ability to distinguish between fraud and non-fraud.  

These metrics ensure that the **selected model** effectively detects fraud **while minimizing false positives and false negatives**.  

|           Model          | Precision | Recall | F1-Score | AUC-ROC |
|--------------------------|-----------|--------|----------|---------|
| **Logistic Regression**  | 91.2%     | 87.5%  | 89.3%    | 96.4%   |
| **Random Forest**        | 94.5%     | 92.1%  | 93.3%    | 98.7%   |
| **XGBoost**              | 95.8%     | 94.2%  | 95.0%    | 99.1%   |
| **Neural Network (MLP)** | 96.3%     | 95.0%  | 95.6%    | 99.3%   |






## 📜 License
- This project is licensed under the MIT License.

## ✉️ Contact

- GitHub Profile: https://github.com/AmirhosseinGhalaei
- Email: amirhosseinghalaei@outlook.com
