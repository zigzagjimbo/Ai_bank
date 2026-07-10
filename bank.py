import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Dataset Exploration
# ==========================================

dataset = pd.read_csv("loan_dataset_20000.csv")

print("Dataset Dimensions:")
print(dataset.shape)

print("\nSample Records:")
print(dataset.head())

print("\nGeneral Information:")
dataset.info()

print("\nMissing Data:")
print(dataset.isna().sum())

print("\nClass Distribution:")
print(dataset["loan_paid_back"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=dataset, x="loan_paid_back")
plt.title("Loan Paid Back Distribution")
plt.show()

# ==========================================
# Feature Engineering
# ==========================================

features = dataset.drop(columns=["loan_paid_back"])
target = dataset["loan_paid_back"]

features = pd.get_dummies(features, drop_first=True)

print("\nEncoded Dataset Shape:")
print(features.shape)

# ==========================================
# Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42,
    stratify=target
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# ==========================================
# Feature Scaling
# ==========================================

normalizer = MinMaxScaler()

X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# ==========================================
# Model Definitions
# ==========================================

classifiers = {}

classifiers["SVM"] = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

classifiers["Logistic Regression"] = LogisticRegression(
    max_iter=1000,
    random_state=42
)

classifiers["Decision Tree"] = DecisionTreeClassifier(
    random_state=42
)

classifiers["Random Forest"] = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================
# Training & Evaluation
# ==========================================

evaluation_results = []

for model_name in classifiers:

    model = classifiers[model_name]

    print("\n" + "-" * 65)
    print(f"Model : {model_name}")
    print("-" * 65)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    matrix = confusion_matrix(y_test, predictions)

    evaluation_results.append(
        [model_name, acc, prec, rec, f1]
    )

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    print("\nConfusion Matrix")
    print(matrix)

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

# ==========================================
# Performance Comparison
# ==========================================

comparison_table = pd.DataFrame(
    evaluation_results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 70)
print("MODELS PERFORMANCE SUMMARY")
print("=" * 70)

print(comparison_table)

# ==========================================
# Accuracy Visualization
# ==========================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=comparison_table,
    x="Model",
    y="Accuracy"
)

plt.title("Accuracy Comparison Between Models")
plt.ylim(0, 1)

for index, score in enumerate(comparison_table["Accuracy"]):
    plt.text(
        index,
        score + 0.01,
        f"{score:.3f}",
        ha="center"
    )

plt.show()

# ==========================================
# F1 Score Visualization
# ==========================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=comparison_table,
    x="Model",
    y="F1 Score"
)

plt.title("F1 Score Comparison Between Models")
plt.ylim(0, 1)

for index, score in enumerate(comparison_table["F1 Score"]):
    plt.text(
        index,
        score + 0.01,
        f"{score:.3f}",
        ha="center"
    )

plt.show()

# ==========================================
# Correlation Analysis
# ==========================================

numeric_data = dataset.select_dtypes(include=np.number)

plt.figure(figsize=(15, 10))

sns.heatmap(
    numeric_data.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()

# ==========================================
# Best Performing Model
# ==========================================

top_model = comparison_table.loc[
    comparison_table["Accuracy"].idxmax()
]

print("\nBest Model According To Accuracy:")
print(top_model)