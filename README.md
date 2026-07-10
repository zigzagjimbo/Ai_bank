# AIBank — Bank Loan Approval Prediction

A binary classification project that predicts whether a bank loan application should be approved, using four different machine learning models trained on a 20,000-record dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

## Overview

This project builds and compares four classification models to predict loan approval outcomes from applicant data. The goal is to identify which algorithm generalizes best on this dataset and to understand the trade-offs between model complexity, interpretability, and accuracy.

**Models implemented:**
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- Random Forest

## Dataset

- **Size:** 20,000 records
- **Task:** Binary classification (loan approved / not approved)
- **Features:** Applicant and loan-related attributes (categorical + numerical)

> Note: add the dataset source/link here (e.g. Kaggle) if it's public, or a short note if it's private/course-provided.

## Pipeline

1. **Data Preprocessing**
   - Handling missing values
   - One-Hot Encoding for categorical features
   - MinMaxScaler for numerical feature normalization
2. **Train/Test Split**
   - Data split into training and testing sets for unbiased evaluation
3. **Model Training**
   - Four classifiers trained independently: SVM, Logistic Regression, Decision Tree, Random Forest
4. **Evaluation**
   - Models compared using accuracy, precision, recall, F1-score, and confusion matrix

## Results

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | —        | —         | —      | —        |
| Decision Tree       | —        | —         | —      | —        |
| Random Forest       | —        | —         | —      | —        |
| SVM                 | —        | —         | —      | —        |

> Fill this table with your actual metrics — it's the first thing recruiters look at, so don't leave it blank in the final version.

## Project Structure

```
aibank/
├── data/                # raw and processed dataset (or a note on where to get it)
├── notebooks/           # Jupyter notebooks for exploration and experiments
├── src/                 # preprocessing and model training scripts
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate.py
├── results/             # saved metrics, plots, confusion matrices
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/aibank.git
cd aibank

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python src/train_models.py
```

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn (for visualizations)

## Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation for more robust evaluation
- Feature importance analysis
- Handling class imbalance (if present)

## Author

**Sadra** — Computer Engineering Student
