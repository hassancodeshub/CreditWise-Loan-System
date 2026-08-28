# 💳 CreditWise Loan Approval System

A Machine Learning project that predicts whether a loan application should be **Approved or Rejected** based on applicant financial, personal, employment, credit, and loan-related information.

---

## 📌 Project Overview

CreditWise Loan System is designed to support financial institutions in making faster and more consistent loan approval decisions.

Traditional loan evaluation can involve manual verification of income, employment details, credit history, and other applicant information. This process can be time-consuming and may result in inconsistent decisions.

This project uses historical loan application data to build a Machine Learning classification system capable of predicting loan approval outcomes.

The target variable is:

- `1` → Loan Approved
- `0` → Loan Rejected

---

## 🎯 Problem Statement

A financial institution receives hundreds of loan applications from customers across urban and rural regions.

Manual evaluation of applications can be time-consuming and inconsistent. The objective of this project is to develop an intelligent Machine Learning system that can analyze applicant information and predict whether a loan should be approved or rejected before final human verification.

The system aims to provide:

- Faster loan application analysis
- Consistent predictions
- Data-driven decision support
- Identification of patterns in historical applications
- Improved efficiency in the loan evaluation process

---

## 📊 Dataset

The dataset contains **1,000 loan application records** with information about applicants and their loan requests.

### Features

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique applicant identifier |
| `Applicant_Income` | Monthly income of applicant |
| `Coapplicant_Income` | Monthly income of co-applicant |
| `Employment_Status` | Employment category |
| `Age` | Applicant age |
| `Marital_Status` | Marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan duration in months |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Urban, Semi-Urban, or Rural |
| `Education_Level` | Education qualification |
| `Gender` | Applicant gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target variable |

---

## 🔎 Exploratory Data Analysis

The project performs Exploratory Data Analysis (EDA) to understand the dataset and identify important patterns.

EDA includes:

- Dataset structure and information
- Missing-value analysis
- Descriptive statistics
- Loan approval class distribution
- Education-level distribution
- Applicant income distribution
- Co-applicant income distribution
- Outlier detection
- Credit score analysis
- DTI ratio analysis
- Savings analysis
- Correlation analysis

Visualizations are created using **Matplotlib** and **Seaborn**.

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Missing Value Treatment

Numerical columns are handled using mean imputation.

Categorical columns are handled using the most frequent value.

### 2. Remove Applicant ID

`Applicant_ID` is removed because it is an identifier and does not provide meaningful predictive information.

### 3. Encoding

Categorical variables are converted into numerical representations.

- Label Encoding is used for `Education_Level`
- One-Hot Encoding is used for categorical features such as:
  - Employment Status
  - Marital Status
  - Loan Purpose
  - Property Area
  - Gender
  - Employer Category

### 4. Feature Scaling

`StandardScaler` is used to standardize numerical features before model training.

---

## 🛠️ Feature Engineering

Additional features are created to improve the representation of important financial relationships.

The project creates:

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
