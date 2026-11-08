"""Train five classifiers on the UCI Dry Bean dataset and save models + metrics.

Run from the project folder:  python train_models.py
Outputs: model/*.joblib, model/metrics.csv, test_data.csv
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics_utils import compute_metrics

SEED = 42
TARGET = "Class"
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    bean = fetch_ucirepo(id=602)  # Dry Bean Dataset
    df = bean.data.features.copy()
    df[TARGET] = bean.data.targets.iloc[:, 0].values
    return df


def build_candidates():
    """Each entry: (display name, file stem, pipeline, hyper-parameter grid)."""
    return [
        (
            "Logistic Regression",
            "logistic_regression",
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
            {"clf__C": [0.1, 1, 10]},
        ),
        (
            "Decision Tree",
            "decision_tree",
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [5, 8, 12], "clf__min_samples_leaf": [1, 5, 10]},
        ),
        (
            "KNN",
            "knn",
            Pipeline([("scale", StandardScaler()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 9, 15], "clf__weights": ["uniform", "distance"]},
        ),
        (
            "Naive Bayes",
            "naive_bayes",
            Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        (
            "Support Vector Machine",
            "svm",
            Pipeline([("scale", StandardScaler()),
                      ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))]),
            {"clf__C": [1, 10], "clf__gamma": ["scale"]},
        ),
    ]


def main():
    df = load_dataset()
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1] - 1} features")

    encoder = LabelEncoder().fit(df[TARGET])
    X = df.drop(columns=TARGET)
    y = encoder.transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Test CSV keeps the original string labels, as a user would upload it.
    test_df = X_test.copy()
    test_df[TARGET] = encoder.inverse_transform(y_test)
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.joblib")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for name, stem, pipe, grid in build_candidates():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        joblib.dump(best, MODEL_DIR / f"{stem}.joblib")

        scores = compute_metrics(y_test, best.predict(X_test), best.predict_proba(X_test))
        rows.append({"ML Model Name": name, "File": f"{stem}.joblib", **scores})
        print(f"{name:25s} best={search.best_params_}  {scores}")

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\n", metrics.to_string(index=False))


if __name__ == "__main__":
    main()
