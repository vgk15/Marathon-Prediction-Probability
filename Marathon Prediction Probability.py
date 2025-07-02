#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 15:31:23 2025

@author: vgkamagere15
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from scipy.interpolate import make_interp_spline

# Load the dataset
df = pd.read_csv("Dataset-Boston-2019.csv")

# Data Cleaning
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Rank_Tot'] = pd.to_numeric(df['Rank_Tot'], errors='coerce')
df = df.dropna(subset=['Age', 'Rank_Tot'])


# Create binary label: Top 25% finishers
top25_cutoff = df['Rank_Tot'].quantile(0.25)
df['Top25'] = (df['Rank_Tot'] <= top25_cutoff).astype(int)

# Feature Engineering
df['Age_bucket'] = pd.cut(df['Age'], bins=[10, 20, 30, 40, 50, 60, 70, 90])
X = pd.get_dummies(df[['Age', 'Age_bucket']], drop_first=True)
y = df['Top25']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Model Evaluation
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {acc:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Best Parameters: {grid.best_params_}")

# ROC Curve Visualization
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(10, 7))
sns.set(style="whitegrid")
plt.plot(fpr, tpr, color='darkblue', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Chance')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve: Predicting Top 25% Marathon Finishers by Age', fontsize=16, weight='bold')
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# Age-specific Marathon Finishers Probability Plot
age_range = np.arange(15, 80)
age_df = pd.DataFrame({'Age': age_range})
age_df['Age_bucket'] = pd.cut(age_df['Age'], bins=[20, 30, 40])
age_encoded = pd.get_dummies(age_df, drop_first=True)
missing_cols = set(X.columns) - set(age_encoded.columns)
for col in missing_cols:
    age_encoded[col] = 0
age_encoded = age_encoded[X.columns]
age_df['Top25_Prob'] = best_model.predict_proba(age_encoded)[:, 1]

# Smoothed Probability Curve
xnew = np.linspace(age_df['Age'].min(), age_df['Age'].max(), 300)
spline = make_interp_spline(age_df['Age'], age_df['Top25_Prob'], k=3)
y_smooth = spline(xnew)

plt.figure(figsize=(10, 7))
plt.plot(xnew, y_smooth, color='blue', linewidth= 3, label='Predicted Probability')
plt.fill_between(xnew, y_smooth, alpha=0.15, color='purple')
plt.title('Top 25% Finish Probability by Age', fontsize=16, weight='bold')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Predicted Probability', fontsize=14)
plt.ylim(0, 1)
plt.xticks(np.arange(15, 81, 5))
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("age_probability.png")
plt.show()
