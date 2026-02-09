import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("../data/campus_placement.csv")
print(f"Dataset shape: {df.shape}\n")

# --------------------------------------------------
# DROP NON-REQUIRED COLUMN
# --------------------------------------------------
# Salary is not used (regression)
df = df.drop(columns=["salary_lpa"], errors="ignore")

# --------------------------------------------------
# TARGET & FEATURES
# --------------------------------------------------
X = df.drop("placed", axis=1)
y = df["placed"]

# --------------------------------------------------
# CATEGORICAL & NUMERICAL COLUMNS
# --------------------------------------------------
categorical_cols = [
    "city_tier",
    "ssc_board",
    "hsc_board",
    "hsc_stream",
    "degree_field",
    "specialization"
]

numerical_cols = [col for col in X.columns if col not in categorical_cols]

# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])

# --------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}\n")

# --------------------------------------------------
# MODELS
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# --------------------------------------------------
# TRAIN & EVALUATE
# --------------------------------------------------
results = []

X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_np, y_train_np)
    y_pred = model.predict(X_test_np)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_np)[:, 1]
        auc = roc_auc_score(y_test_np, y_prob)
    else:
        auc = None

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test_np, y_pred), 4),
        "AUC": round(auc, 4) if auc is not None else "N/A",
        "Precision": round(precision_score(y_test_np, y_pred), 4),
        "Recall": round(recall_score(y_test_np, y_pred), 4),
        "F1 Score": round(f1_score(y_test_np, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test_np, y_pred), 4)
    })

# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE – CAMPUS PLACEMENT PREDICTION")
print("=" * 80)
print(results_df.to_string(index=False))
