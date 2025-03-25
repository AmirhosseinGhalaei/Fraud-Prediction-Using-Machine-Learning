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
- The dataset utilized in this project is the Credit Card Fraud Detection dataset, sourced from the ULB Machine Learning Group and made available on Kaggle. 
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
- This dataset comprises transactions made by European cardholders over a two-day period in September 2013. Out of 284,807 transactions, 492 are labeled as fraudulent, highlighting the dataset's highly imbalanced nature, with fraudulent transactions accounting for only 0.172% of the data.
Due to confidentiality concerns, the dataset's features have undergone principal component analysis (PCA) transformation, resulting in 28 anonymized features labeled V1 through V28. Additionally, the dataset includes the 'Time' feature, representing the seconds elapsed between the first transaction and each subsequent transaction, and the 'Amount' feature, indicating the transaction amount. The target variable, 'Class,' indicates the transaction type: '0' for legitimate transactions and '1' for fraudulent ones.
- This dataset is widely used for benchmarking fraud detection systems and presents challenges typical of real-world scenarios, such as class imbalance and the need for anonymized feature interpretation.


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

## 🔹 7.1 Model Selection & Training  
We experiment with various machine learning algorithms, including:  

✅ **Logistic Regression** – A baseline model for interpretability.  
✅ **Random Forest** – A robust ensemble model that captures complex relationships.  
✅ **XGBoost** – A gradient boosting model known for handling imbalanced datasets effectively.  
✅ **Neural Networks (MLP Classifier)** – A deep learning-based approach for improved pattern recognition.  

Each model is trained on the **preprocessed dataset** using **stratified k-fold cross-validation** to ensure balanced class distribution during training.  

## 🔹 7.2 Model Evaluation Metrics  
Given the severe **class imbalance**, traditional **accuracy** is not an ideal metric. Instead, we focus on:  

🎯 **Precision** – Minimizing false positives.  
🎯 **Recall** – Ensuring fraudulent transactions are detected.  
🎯 **F1-Score** – A balance between precision and recall.  
🎯 **AUC-ROC Score** – Measures overall model performance in distinguishing fraud from non-fraud.  

These metrics provide a **comprehensive evaluation** of our models, ensuring that we prioritize **fraud detection accuracy** while minimizing false alarms.  




## 📜 License
- This project is licensed under the MIT License.

## ✉️ Contact

- GitHub Profile: https://github.com/AmirhosseinGhalaei
- Email: amirhosseinghalaei@outlook.com
