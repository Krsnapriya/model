import json
import os

nb_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Pediatric Disease Prediction: Multi-Model Comparison\n",
    "\n",
    "This notebook trains and compares three models (Logistic Regression, Random Forest, XGBoost) to predict pediatric diseases:\n",
    "0. Healthy\n",
    "1. Sepsis\n",
    "2. Pneumonia\n",
    "3. Diabetes Type 1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import xgboost as xgb\n",
    "import shap\n",
    "import joblib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Multi-Class Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../data/pediatric_ehr_synthetic.csv')\n",
    "print(\"Shape:\", df.shape)\n",
    "print(\"Class Balance:\\n\", df['Target_Label'].value_counts(normalize=True))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop('Target_Label', axis=1)\n",
    "y = df['Target_Label']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "# Scale data for Logistic Regression\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Model Training & Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    \"Logistic Regression\": LogisticRegression(max_iter=1000),\n",
    "    \"Random Forest\": RandomForestClassifier(n_estimators=100, random_state=42),\n",
    "    \"XGBoost\": xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', use_label_encoder=False)\n",
    "}\n",
    "\n",
    "results = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    print(f\"Training {name}...\")\n",
    "    if name == \"Logistic Regression\":\n",
    "        model.fit(X_train_scaled, y_train)\n",
    "        preds = model.predict(X_test_scaled)\n",
    "    else:\n",
    "        model.fit(X_train, y_train)\n",
    "        preds = model.predict(X_test)\n",
    "        \n",
    "    acc = accuracy_score(y_test, preds)\n",
    "    f1 = f1_score(y_test, preds, average='weighted')\n",
    "    results[name] = {\"Accuracy\": acc, \"F1\": f1}\n",
    "    print(f\"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}\")\n",
    "\n",
    "results_df = pd.DataFrame(results).T\n",
    "results_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Evaluation of Best Model (XGBoost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = models[\"XGBoost\"]\n",
    "y_pred = best_model.predict(X_test)\n",
    "\n",
    "print(classification_report(y_test, y_pred, target_names=['Healthy', 'Sepsis', 'Pneumonia', 'Diabetes']))\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Sepsis', 'Pneumonia', 'Diabetes'], yticklabels=['Healthy', 'Sepsis', 'Pneumonia', 'Diabetes'])\n",
    "plt.title(\"Confusion Matrix\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. SHAP Explainability"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "explainer = shap.TreeExplainer(best_model)\n",
    "shap_values = explainer.shap_values(X_test)\n",
    "\n",
    "# Summary plot for multi-class\n",
    "shap.summary_plot(shap_values, X_test, class_names=['Healthy', 'Sepsis', 'Pneumonia', 'Diabetes'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Save Artifacts for Dashboard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "joblib.dump(best_model, '../api/pediatric_model.joblib')\n",
    "joblib.dump(scaler, '../api/scaler.joblib') # Save scaler just in case, though XGB didn't use it\n",
    "print(\"Model saved to ../api/pediatric_model.joblib\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('notebooks/pediatric_prediction_prototype.ipynb', 'w') as f:
    json.dump(nb_content, f, indent=4)

print("Enhanced Notebook created.")
