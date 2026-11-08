"""Dry Bean classifier explorer - Streamlit front end.

Loads the pre-trained pipelines from ./model and evaluates them on an uploaded
test CSV. Nothing is retrained here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import METRIC_COLUMNS, compute_metrics

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    features = joblib.load(MODEL_DIR / "feature_names.joblib")
    return models, encoder, features


@st.cache_data
def evaluate(df: pd.DataFrame, model_name: str):
    """Score one saved model on the given frame (cached per frame + model)."""
    models, encoder, features = load_artifacts()
    model = models[model_name]
    y_true = encoder.transform(df[TARGET])
    X = df[features]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    scores = compute_metrics(y_true, y_pred, y_proba, labels=np.arange(len(encoder.classes_)))
    return y_true, y_pred, scores


models, encoder, features = load_artifacts()
class_names = list(encoder.classes_)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None,
                             disabled=uploaded is not None)
    st.header("2 · Model")
    chosen = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.caption(f"Required columns: {len(features)} features + `{TARGET}`")

st.title("🫘 Dry Bean Classifier Lab")
st.write(
    "Pick a pre-trained model, feed it a test CSV and inspect how well it "
    "separates the **7 dry bean varieties**. Models are loaded from disk; "
    "your file is only used for scoring."
)

# ------------------------------------------------------------- load data
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    data = pd.read_csv(ROOT / "test_data.csv")
    source = "test_data.csv (bundled)"
else:
    st.info("Upload a test CSV or tick the bundled sample in the sidebar.")
    st.stop()

missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

data = data.dropna(subset=features + [TARGET])
unknown = sorted(set(data[TARGET]) - set(class_names))
if unknown:
    st.error(f"Unknown class labels in `{TARGET}`: {unknown}. Expected: {class_names}")
    st.stop()

st.success(f"Loaded **{source}** - {len(data):,} rows")
with st.expander("Preview data"):
    st.dataframe(data.head(20), width="stretch")

# ----------------------------------------------------------- selected model
y_true, y_pred, scores = evaluate(data, chosen)

st.subheader(f"Results · {chosen}")
cols = st.columns(len(METRIC_COLUMNS))
for col, metric in zip(cols, METRIC_COLUMNS):
    col.metric(metric, "n/a" if np.isnan(scores[metric]) else f"{scores[metric]:.4f}")
st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

# ------------------------------------------------------ all-model comparison
st.subheader("All models on this data")
comparison = pd.DataFrame(
    [{"ML Model Name": n, **evaluate(data, n)[2]} for n in MODEL_FILES]
).set_index("ML Model Name")
st.dataframe(
    comparison.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
    width="stretch",
)
best = comparison["MCC"].idxmax()
st.caption(f"Highest MCC on this file: **{best}**")
st.bar_chart(comparison[["Accuracy", "F1", "MCC"]])
