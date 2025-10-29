💳 Credit Card Fraud Detection using Machine Learning & PyCaret
📘 Overview

This project focuses on detecting fraudulent credit card transactions using both traditional machine learning techniques and automated ML with PyCaret.

The dataset used is the Kaggle Credit Card Fraud Detection Dataset, which is highly imbalanced, containing a very small percentage of fraud cases.

To handle this challenge, the project is implemented in two complementary approaches:
1️⃣ Manual ML Pipeline: Handling data imbalance using techniques like SMOTE, ADASYN, and Random Under/Oversampling, followed by model training and evaluation.
2️⃣ PyCaret Approach: Leveraging PyCaret’s Classification and Anomaly Detection modules to automatically compare multiple models (both supervised and unsupervised) for fraud detection.

⚙️ Tech Stack

Language: Python

Platform: Jupyter Notebook

Dataset: Kaggle – Credit Card Fraud Detection

Libraries Used:

🧩 Machine Learning: scikit-learn, imbalanced-learn, xgboost, lightgbm, catboost

⚡ AutoML: pycaret

📊 Visualization: matplotlib, seaborn, plotly

🧮 Data Handling: pandas, numpy

🧠 Project Approaches
1️⃣ Manual Machine Learning Workflow
Steps:

Performed data cleaning, EDA, and visualized the class imbalance.

Applied feature scaling and train-test split.

Used different resampling strategies to handle imbalance:

SMOTE (Synthetic Minority Oversampling)

ADASYN (Adaptive Synthetic Sampling)

Random Oversampling / Undersampling

Trained multiple classification models:

Logistic Regression

Decision Tree

Random Forest

XGBoost

LightGBM

CatBoost

SVM

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Curve

📈 Special focus was given to maximizing Recall (since missing a fraud is costlier than a false alarm).

2️⃣ Automated Approach using PyCaret

PyCaret simplifies the entire ML workflow — from setup to model comparison — in a few lines of code.

(a) Supervised Model Comparison
from pycaret.classification import *
clf = setup(data=df, target='Class', session_id=123)
best_model = compare_models()


📊 PyCaret automatically trains and ranks multiple classification models (Logistic Regression, Random Forest, LightGBM, XGBoost, etc.) based on metrics like AUC, F1, and Precision-Recall.

(b) Unsupervised / Anomaly Detection

To explore fraud detection without labels:

from pycaret.anomaly import *
anom = setup(data=df, session_id=123)
iforest = create_model('iforest')     # Isolation Forest
results = assign_model(iforest)
plot_model(iforest, plot='tsne')


🧩 Models used:

Isolation Forest

DBSCAN

KMeans

One-Class SVM

PCA-based Detection

Anomalies identified by these models were compared visually using t-SNE and UMAP plots.

🏆 Results & Insights
| Approach                      | Key Highlights                                                  | Best Performing Model    |
| ----------------------------- | --------------------------------------------------------------- | ------------------------ |
| **Manual ML (Balanced Data)** | Improved minority class detection after SMOTE                   | XGBoost / LightGBM       |
| **PyCaret (Supervised)**      | Auto-tuning gave similar or higher recall with less manual work | Random Forest / CatBoost |
| **PyCaret (Unsupervised)**    | Isolation Forest effectively isolated fraud cases               | Isolation Forest         |


✅ Balancing techniques (especially SMOTE) significantly improved recall.
✅ PyCaret reduced modeling time drastically with automated comparisons.
✅ Unsupervised models offered additional insight into potential unknown fraud patterns.

📈 Visualizations

Correlation heatmap and class imbalance plot

ROC Curves and Precision-Recall plots

Feature importance visualization

t-SNE anomaly detection map

