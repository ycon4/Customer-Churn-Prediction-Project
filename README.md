# Customer Churn Prediction in Digital Banking

A machine learning project that predicts which customers of a digital bank are at high risk of churning. The notebook compares **k-Nearest Neighbors**, **Logistic Regression**, and **Linear SVM** (each with L1 and L2 regularization where applicable) and reports which model and which predictors work best.

Developed by **Group 3 (IT3D.1)** as a course project.

## Problem Statement

A digital bank is experiencing fluctuating retention rates. With growing competition from fintech startups and changing user behavior, the bank wants to identify customers at risk of leaving the platform or becoming inactive, so that retention efforts can be targeted early.

**Objectives**

- Analyze the features that contribute to customer churn
- Classify customers by churn risk using machine learning
- Provide insights the bank can use to reduce churn and improve engagement

## Dataset

`churn_risk_dataset.csv` contains **1,000 customer records** with no missing values. Labels were assigned using business-defined rules: **650 customers are high risk (1)** and **350 are low risk (0)**.

| Feature | Description |
| ------- | ----------- |
| `age` | Customer age |
| `income` | Income |
| `marital_status` | Single, married, or widowed (one-hot encoded) |
| `dependents` | Number of dependents |
| `employment_years` | Years of employment |
| `credit_score` | Credit score |
| `debt_to_income_ratio` | Debt-to-income ratio |
| `num_credit_cards` | Number of credit cards |
| `savings_balance` | Savings balance |
| `credit_usage` | Credit usage |
| `avg_monthly_spending` | Average monthly spending |
| `internet_usage_hours` | Internet usage in hours |
| `social_media_usage_hours` | Social media usage in hours |
| `churn_risk` | **Target:** 1 = high risk, 0 = low risk |

## Methodology

1. **Exploratory data analysis:** summary statistics, feature histograms, target distribution, missing-value check, and a correlation heatmap
2. **Preprocessing:** one-hot encoding of `marital_status` and standardization of features with `StandardScaler`
3. **Train-test split selection:** test sizes of 0.20, 0.25, 0.30, and 0.35 were each evaluated across 50 random states using stratified splits, and the best-performing configuration was reused for all models
4. **Modeling:**
   - **kNN:** top 10 features selected with Random Forest and Recursive Feature Elimination (RFE), then K tuned from 1 to 99
   - **Logistic Regression:** L1 and L2 penalties, with C values compared manually and tuned with `GridSearchCV`
   - **Linear SVM (LinearSVC):** L1 and L2 penalties across a range of C values, with coefficient plots to inspect feature importance
5. **Evaluation:** training and test accuracy, classification reports, and top predictor variables

## Results

| Model | Test Size | Training Accuracy | Test Accuracy | Best Parameter | Top Predictor |
| :---: | :---: | :---: | :---: | :---: | :---: |
| kNN | 25% | 99.26% | 99% | k = 1 | N/A |
| Logistic Regression (L1) | 25% | 98.73% | 98.83% | C = 10 | `age` |
| Logistic Regression (L2) | 25% | 98.60% | 98.83% | C = 10 | `age` |
| Linear SVM (L1) | 25% | 96.88% | 96.50% | C = 0.01 | `internet_usage_hours` |
| Linear SVM (L2) | 25% | 98.00% | 99.50% | C = 0.01 | `avg_monthly_spending` |

**Key findings**

- **Linear SVM with L2 regularization** achieved the highest test accuracy (99.50%).
- kNN and Logistic Regression performed similarly, but kNN offers no feature-level interpretability.
- Regularized linear models give the best balance of accuracy and interpretability.
- Influential predictors include `age`, `internet_usage_hours`, and `avg_monthly_spending`.

## Repository Contents

```
.
├── churn_risk_dataset.csv                                       Dataset (1,000 customers)
└── group-3_Logistic-Regression-and-SVM-exercises-IT3D.1_Finalized.ipynb   Full analysis notebook
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/ycon4/Customer-Churn-Prediction-ML.git
cd Customer-Churn-Prediction-ML

pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

### Running the Notebook

```bash
jupyter notebook group-3_Logistic-Regression-and-SVM-exercises-IT3D.1_Finalized.ipynb
```

Run the cells from top to bottom. Keep `churn_risk_dataset.csv` in the same folder as the notebook, since it is loaded with a relative path.

## Notes and Limitations

- The churn labels come from business-defined rules rather than observed customer behavior, so the very high accuracies mostly reflect how well the models recover those rules. Results on real churn data would likely be lower.
- The dataset is small (1,000 records), and the best kNN result at k = 1 suggests that nearby records are very similar.

## Authors

Group 3 (IT3D.1)
Repository maintained by Andrei G. Raagas
