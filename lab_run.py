"""Self-contained run of the whole Dry Bean experiment (for BITS Virtual Lab).

Needs: pandas, scikit-learn, ucimlrepo (pip install ucimlrepo).
Usage: python lab_run.py
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

bean = fetch_ucirepo(id=602)
X, y = bean.data.features, bean.data.targets.iloc[:, 0]
print("Dataset:", X.shape[0], "rows,", X.shape[1], "features,", y.nunique(), "classes")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
    "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=42)),
}

rows = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred, proba = model.predict(X_te), model.predict_proba(X_te)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_te, pred, average="macro"),
        "Recall": recall_score(y_te, pred, average="macro"),
        "F1": f1_score(y_te, pred, average="macro"),
        "MCC": matthews_corrcoef(y_te, pred),
    })

print(pd.DataFrame(rows).round(4).to_string(index=False))
