
# ================================= Import Libraries =================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ================================= Data Preprocessing =================================

# Load the dataset from the same directory
df = pd.read_csv("creditcard.csv")

# Inspect the dataset
print("Dataset Shape:", df.shape)  # Number of rows & columns
print("\nColumn Info:")
print(df.info())  # Data types & missing values

# Handle missing values (Fill missing numerical values with median)
print("Missing Values Per Column:")
print(df.isnull().sum())

# Define features (X) and target variable (y)
X = df.drop(columns=['Class'])  # All features except the target
y = df['Class']  # Target variable

# Split into training (80%) and validation (20%) sets before oversampling
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution before oversampling
print("Class distribution before oversampling:")
print(y_train.value_counts())

# Initialize oversampler
oversampler = RandomOverSampler(random_state=42)

# Apply oversampling only on the training set
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Check class distribution after resampling
print("Class distribution after oversampling:")
print(y_train_resampled.value_counts())

# Initialize StandardScaler
scaler = StandardScaler()

# Scale features **after** splitting to prevent data leakage
X_train_resampled[['Time', 'Amount']] = scaler.fit_transform(X_train_resampled[['Time', 'Amount']])
X_val[['Time', 'Amount']] = scaler.transform(X_val[['Time', 'Amount']])

# Verify scaling
print(X_train_resampled[['Time', 'Amount']].describe())

# ================================= Feature Selection & Engineering =================================

# Correlation Analysis
correlation_matrix = X_train_resampled.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_resampled.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_resampled.values, i) for i in range(X_train_resampled.shape[1])]

# Display VIF values
print(vif_data.sort_values(by="VIF", ascending=False))

# List of features to drop
high_vif_features = ["V7", "V3", "V17", "V16", "V12", "V5", "V10", "V2", "V14", "V18", "V11", "V1"]

# Drop features with high VIF
X_train_filtered = X_train_resampled.drop(columns=high_vif_features)
X_val_filtered = X_val.drop(columns=high_vif_features)

# Recalculate VIF
vif_data_filtered = pd.DataFrame()
vif_data_filtered["Feature"] = X_train_filtered.columns
vif_data_filtered["VIF"] = [variance_inflation_factor(X_train_filtered.values, i) for i in range(X_train_filtered.shape[1])]

# Display new VIF values
print(vif_data_filtered.sort_values(by="VIF", ascending=False))

# Feature Importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_filtered, y_train_resampled)

# Get feature importance scores
importances = rf.feature_importances_
feature_names = X_train_filtered.columns

# Sort by importance
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 5))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X_train_filtered.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_filtered.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# Final Check
selected_features = ["V4", "V9", "V21", "V6", "V27", "V19", "V8", "V20", "V28", "Amount"]

# Update datasets with selected features
X_train_final = X_train_filtered[selected_features]
X_val_final = X_val_filtered[selected_features]

# Check correlation between selected features and target variable
correlation_matrix = pd.concat([X_train_final, y_train_resampled], axis=1).corr()
print(correlation_matrix["Class"].sort_values(ascending=False))

print(y_train_resampled.value_counts(normalize=True))  # Show class distribution
sns.countplot(x=y_train_resampled)
plt.title("Class Distribution")
plt.show()

# ================================= Model Training & Evaluation =================================

# Define models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# Train & evaluate models
for name, model in models.items():
    model.fit(X_train_final, y_train_resampled)
    y_pred = model.predict(X_val_final)
    y_pred_proba = model.predict_proba(X_val_final)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n {name} Performance:")
    print(f" Accuracy: {round(accuracy, 4)}")
    print(f" AUC Score: {round(auc_score, 4)}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

# ================================= Model Hyperparameter Tuning =================================

# Random Forest with balanced class weight
best_rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                 min_samples_leaf=4, class_weight="balanced", random_state=42)
best_rf.fit(X_train_final, y_train_resampled)

# Compute fraud class weight for XGBoost
fraud_weight = len(y_train_resampled) / sum(y_train_resampled == 1)  # Adjust weight for fraud cases

# XGBoost with scale_pos_weight to handle fraud class imbalance
best_xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                         scale_pos_weight=fraud_weight, eval_metric="logloss", random_state=42)
best_xgb.fit(X_train_final, y_train_resampled)

# ================================= Final Model Training & Evaluation =================================

# Evaluating the best models on the validation set (X_val_final, y_val)

models = {'Random Forest': best_rf, 'XGBoost': best_xgb}
for name, model in models.items():
    y_pred = model.predict(X_val_final)
    y_pred_proba = model.predict_proba(X_val_final)[:, 1]

    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}")

    print("Classification Report:")
    print(classification_report(y_val, y_pred))
