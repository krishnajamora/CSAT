# ------------------------------
# DeepCSAT E-commerce Project Code (Datetime Fix)
# ------------------------------

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import gdown
import warnings
warnings.filterwarnings('ignore')

# Step 2: Download Dataset from Google Drive
url = "https://drive.google.com/uc?id=10pFYAEZqnZ9mQHUrxly7xwe9qRKL7uM5"
output = "DeepCSAT_Ecommerce.csv"
gdown.download(url, output, quiet=False)

# Step 3: Load Dataset
df = pd.read_csv(output)
print(f"Dataset loaded with shape: {df.shape}")

# Step 4: Data Overview
print(df.head())
print("\nData Types & Null Values:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())

# Step 5: Handle Missing Values & Duplicates
df.drop_duplicates(inplace=True)
for col in df.columns:
    if df[col].dtype in ['float64','int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nAfter imputation, missing values:\n", df.isnull().sum())

# Step 6: Define Target Column Correctly with Spaces
target_col = 'CSAT Score'

# Step 7: Basic EDA - Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(df[target_col])
plt.title('Distribution of Customer Satisfaction Score')
plt.show()

# Step 8: Univariate Analysis - Numeric Variables
num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
num_cols.remove(target_col)

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Step 9: Bivariate Analysis - Numeric Vs Target
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[target_col], y=df[col])
    plt.title(f'{col} vs {target_col}')
    plt.show()

# Step 10: Multivariate Analysis - Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols + [target_col]].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 11: Hypothesis Testing

# Hypothesis 1: Delivery time influences satisfaction
if 'connected_handling_time' in df.columns:
    satisfied = df[df[target_col] >= df[target_col].median()]['connected_handling_time']
    unsatisfied = df[df[target_col] < df[target_col].median()]['connected_handling_time']
    t_stat, p_val = stats.ttest_ind(satisfied, unsatisfied)
    print(f"Hypothesis 1: T-test p-value = {p_val} (Delivery time difference between satisfied and unsatisfied customers)")

# Hypothesis 2: Item price affects satisfaction
if 'Item_price' in df.columns:
    corr, p_val = stats.pearsonr(df['Item_price'], df[target_col])
    print(f"Hypothesis 2: Correlation between Item_price and satisfaction = {corr}, p-value = {p_val}")

# Hypothesis 3: Agent Shift affects satisfaction
if 'Agent Shift' in df.columns:
    groups = pd.crosstab(df['Agent Shift'], df[target_col])
    chi2, p, dof, ex = stats.chi2_contingency(groups)
    print(f"Hypothesis 3: Chi-square test between Agent Shift and satisfaction p-value = {p}")

# Step 12: Feature Engineering
if ('order_date_time' in df.columns) and ('Survey_response_Date' in df.columns):
    df['order_date_time'] = pd.to_datetime(df['order_date_time'], dayfirst=True)
    df['Survey_response_Date'] = pd.to_datetime(df['Survey_response_Date'], dayfirst=True)
    df['days_to_survey'] = (df['Survey_response_Date'] - df['order_date_time']).dt.days
    num_cols.append('days_to_survey')

# Encode categorical variables
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if target_col in cat_cols:
    cat_cols.remove(target_col)

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Scale numeric features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 13: Prepare Features and Target
X = df.drop(columns=[target_col])
y = df[target_col]

# Step 14: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 15: Model Training using Random Forest with Grid Search
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best RF Parameters: {grid_search.best_params_}")

# Step 16: Model Evaluation
y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Step 17: Model Explainability using SHAP
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# Step 18: Save Model for Deployment
joblib.dump(best_rf, 'DeepCSAT_Model.pkl')
print("Model saved as DeepCSAT_Model.pkl")

# Step 19: Sample Prediction
sample = X_test.iloc[0:1]
prediction = best_rf.predict(sample)
print(f"Predicted CSAT Score for sample: {prediction[0]}")
