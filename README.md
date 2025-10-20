DeepCSAT – Ecommerce Customer Satisfaction Prediction
This repository contains an end-to-end machine learning project to predict customer satisfaction scores (CSAT) based on e-commerce platform data. The project covers data preprocessing, exploratory data analysis, model training, evaluation, and explainability.

Project Overview
Predict customer satisfaction score (CSAT Score) from transactional, behavioral, and categorical data.

Perform thorough Exploratory Data Analysis (EDA) with visualizations.

Conduct hypothesis testing relevant to business questions.

Train and tune a Random Forest Classifier.

Explain model predictions using SHAP values.

Provide a saved model ready for deployment.

Requirements
Python 3.8 or higher

Recommended: Virtual environment to isolate dependencies

Installation
Clone or download this repository.

Navigate to the project directory.

Create and activate a virtual environment (optional but recommended):

bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install required packages:

bash
pip install -r requirements.txt
Dependencies
pandas

numpy

seaborn

matplotlib

scipy

scikit-learn

shap

joblib

gdown

warnings (built-in)

You can install all at once via:

bash
pip install pandas numpy seaborn matplotlib scipy scikit-learn shap joblib gdown
Usage
Run the main project script to perform data loading, preprocessing, EDA, train & evaluate the model:

bash
python deepcsat_project.py
The script will:

Download dataset automatically from Google Drive.

Print dataset information and missing value stats.

Show histograms, boxplots, and correlation heatmaps.

Perform hypothesis testing.

Train a Random Forest model with hyperparameter tuning.

Output model accuracy and classification report.

Display feature importance using SHAP.

Save the trained model as DeepCSAT_Model.pkl.

Show a sample prediction.
