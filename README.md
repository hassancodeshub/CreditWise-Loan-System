# 💳 LoanLens — Credit Wise Loan Approval System

**LoanLens** is a machine learning classification project that predicts whether a loan application is likely to be **approved or rejected** based on applicant financial, personal, employment, credit, and loan-related information.

The project explores and compares three machine learning algorithms — **Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes** — and further evaluates them after hyperparameter tuning using **5-fold cross-validation**.

---

## 📌 Project Overview

Loan approval decisions depend on several factors such as an applicant's income, credit score, existing loans, savings, debt-to-income ratio, employment status, loan amount, and other personal and financial characteristics.

The goal of LoanLens is to explore these factors and build machine learning models capable of classifying loan applications based on historical application data.

The project follows a complete machine learning workflow:

```text
Data Collection
      ↓
Data Understanding
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Baseline Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Model Comparison
```

> **Note:** This project is developed for educational purposes and should not be used as the sole basis for real-world lending decisions.

---

## 🎯 Objectives

The main objectives of this project are to:

* Understand and explore the loan approval dataset.
* Identify and handle missing values.
* Perform Exploratory Data Analysis (EDA).
* Understand relationships between applicant characteristics and loan approval.
* Perform categorical encoding and feature scaling.
* Create meaningful engineered features.
* Train multiple classification models.
* Compare model performance using Precision, Recall, and F1-score.
* Perform hyperparameter tuning using 5-fold cross-validation.
* Compare baseline and tuned model performance.
* Identify the best-performing tuned model.

---

## 📊 Dataset

The dataset contains **1,000 loan application records** and **20 columns**, covering applicant demographics, financial information, employment details, credit information, and loan characteristics.

### Features

| Feature              | Description                   |
| -------------------- | ----------------------------- |
| `Applicant_ID`       | Unique applicant identifier   |
| `Applicant_Income`   | Applicant's income            |
| `Coapplicant_Income` | Co-applicant's income         |
| `Employment_Status`  | Applicant's employment status |
| `Age`                | Applicant age                 |
| `Marital_Status`     | Applicant's marital status    |
| `Dependents`         | Number of dependents          |
| `Credit_Score`       | Applicant's credit score      |
| `Existing_Loans`     | Number of existing loans      |
| `DTI_Ratio`          | Debt-to-Income ratio          |
| `Savings`            | Applicant's savings           |
| `Collateral_Value`   | Value of collateral           |
| `Loan_Amount`        | Requested loan amount         |
| `Loan_Term`          | Loan duration                 |
| `Loan_Purpose`       | Purpose of the loan           |
| `Property_Area`      | Property area                 |
| `Education_Level`    | Applicant's education level   |
| `Gender`             | Applicant gender              |
| `Employer_Category`  | Employer category             |
| `Loan_Approved`      | Target variable               |

---

## 🔎 Exploratory Data Analysis

The dataset was explored to understand its distributions, class balance, missing values, and relationships between variables.

The EDA includes:

* Dataset structure and statistical summary
* Missing-value analysis
* Loan approval distribution
* Employment status analysis
* Loan purpose analysis
* Applicant income analysis
* Co-applicant income analysis
* Outlier analysis
* Credit score analysis
* Correlation analysis
* Relationship between important features and loan approval

The target distribution contains:

* **702 rejected applications**
* **298 approved applications**

This shows that the target classes are not perfectly balanced.

---

## 🧹 Data Preprocessing

Before building the models, the dataset was prepared through several preprocessing steps.

### Missing Value Handling

The original dataset contains missing values across several numerical and categorical columns. The notebook handles these missing values using:

* **Mean imputation** for numerical features
* **Most-frequent imputation** for categorical features

After imputation, the dataset contains no remaining missing values.

### Removing Applicant ID

`Applicant_ID` was removed from the model features because it is an identifier rather than a meaningful predictive variable.

### Categorical Encoding

Categorical variables were converted into numerical representations so that they could be used by the machine learning algorithms.

One-hot encoding was used for nominal categorical variables, while label encoding was used where appropriate.

### Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

A fixed `random_state=42` was used to make the results reproducible.

The split was also performed using **stratification** so that the distribution of approved and rejected applications remained similar in both sets.

### Feature Scaling

`StandardScaler` was used to standardize numerical model inputs.

For hyperparameter tuning, scaling was incorporated into the model pipelines so that the scaling step is handled within the cross-validation workflow.

---

## 🛠️ Feature Engineering

To allow the models to capture potential non-linear relationships, squared versions of two important features were created:

```python
data["DTI_Ratio_sq"] = data["DTI_Ratio"] ** 2
data["Credit_Score_sq"] = data["Credit_Score"] ** 2
```

These engineered features were introduced to capture possible non-linear effects of:

* Debt-to-Income Ratio
* Credit Score

---

## 🤖 Machine Learning Models

Three classification algorithms were selected for comparison.

### 1. Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of an observation belonging to a particular class.

It provides a strong and interpretable baseline for binary classification.

### 2. K-Nearest Neighbors (KNN)

KNN is a distance-based classification algorithm.

It predicts the class of an observation based on the classes of its nearest neighboring observations.

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It assumes that continuous features follow a Gaussian distribution within each class.

---

## 📏 Model Evaluation

The models were evaluated using three main classification metrics.

### Precision

Precision measures how many observations predicted as positive were actually positive.

$$
Precision = \frac{TP}{TP + FP}
$$

### Recall

Recall measures how many actual positive observations were correctly identified.

$$
Recall = \frac{TP}{TP + FN}
$$

### F1-Score

F1-score provides a balance between precision and recall.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

Because the target classes are not perfectly balanced, **F1-score** was used as the primary metric during hyperparameter tuning.

---

## 📈 Baseline Model Results

The baseline models were evaluated on the unseen test set.

| Model                   |  Precision |     Recall |   F1-Score |
| ----------------------- | ---------: | ---------: | ---------: |
| **Logistic Regression** | **0.7903** | **0.8033** | **0.7967** |
| KNN                     |     0.6200 |     0.5082 |     0.5586 |
| Naive Bayes             |     0.7833 |     0.7705 |     0.7769 |

### 🏆 Best Baseline Model

**Logistic Regression** achieved the best baseline performance:

* Precision: **0.7903**
* Recall: **0.8033**
* F1-Score: **0.7967**

---

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**.

The purpose of tuning was to search for better model configurations rather than simply relying on the default model parameters.

### Logistic Regression

The following parameters were tuned:

* `C`
* `solver`
* `class_weight`

### KNN

The following parameters were tuned:

* `n_neighbors`
* `weights`
* `metric`

### Gaussian Naive Bayes

The following parameter was tuned:

* `var_smoothing`

The primary optimization metric was:

```python
scoring="f1"
```

---

## 📊 Before vs After Hyperparameter Tuning

| Model               | Before Precision | After Precision | Before Recall | After Recall | Before F1 | After F1 |
| ------------------- | ---------------: | --------------: | ------------: | -----------: | --------: | -------: |
| Logistic Regression |            0.790 |           0.679 |         0.803 |    **0.869** | **0.797** |    0.763 |
| KNN                 |            0.620 |           0.577 |         0.508 |        0.492 |     0.559 |    0.531 |
| Naive Bayes         |            0.783 |           0.783 |         0.770 |        0.770 |     0.777 |    0.777 |

### Key Observations

Hyperparameter tuning did **not consistently improve the test-set performance**.

* **Logistic Regression:** Recall increased from `0.803` to `0.869`, but precision decreased from `0.790` to `0.679`. As a result, its F1-score decreased.
* **KNN:** Precision, recall, and F1-score all decreased after tuning.
* **Naive Bayes:** Performance remained unchanged across all three metrics.

This highlights an important machine learning concept: **hyperparameter tuning does not guarantee better performance on unseen data**.

---

## 🏆 Final Model Selection

Among the **tuned models**, Naive Bayes achieved the highest F1-score:

> **Naive Bayes — F1-Score: 0.777**

Therefore, **Naive Bayes was selected as the best-performing tuned model based on F1-score**.

However, the baseline Logistic Regression model achieved a higher test F1-score of **0.7967**.

This means that, for this particular dataset and experimental setup, hyperparameter tuning did not improve the final test-set F1-score.

Rather than treating this as a failure, the result provides a useful comparison between baseline and tuned models and shows how model performance can change after optimization.

---

## 💡 Key Learnings

Through this project, the following concepts were explored:

* Data cleaning and preprocessing
* Missing-value treatment
* Categorical encoding
* Feature engineering
* Feature scaling
* Exploratory Data Analysis
* Classification algorithms
* Precision, Recall, and F1-score
* Stratified train-test splitting
* Cross-validation
* GridSearchCV
* Hyperparameter tuning
* Model comparison
* Interpretation of model performance

---

## 🧰 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📁 Project Structure

```text
LoanLens/
│
├── LoanLens.ipynb
├── loan_approval_data.csv
├── README.md
└── CreditWise Loan System.pdf
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project folder

```bash
cd LoanLens
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
loan_approval_prediction.ipynb
```

Run the notebook cells sequentially.

---

## 👨‍💻 Author

**Hassan Shah**

**LoanLens — Credit Wise Loan Approval System**

A Machine Learning minor project focused on applying the complete classification workflow, from data preprocessing and exploratory analysis to feature engineering, model training, hyperparameter tuning, evaluation, and model comparison.
