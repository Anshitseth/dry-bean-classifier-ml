# Dry Bean Classification - ML Assignment 2 (M.Tech AIML, BITS WILP)

## a. Problem statement

Classify a dry bean into one of **seven registered varieties** (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) using only the geometric measurements taken from a photograph of the bean. Automating this removes manual sorting and gives a consistent quality-control step for seed and food suppliers.

Five classical classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same train/test split, compared on six metrics, and exposed through an interactive Streamlit app.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository - [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (id 602) |
| Task | Multi-class classification (7 classes) |
| Instances | 13,611 (requirement: >= 500) |
| Features | 16 numeric (requirement: >= 12) |
| Missing values | None |
| Target | `Class` |
| Split | 80 / 20 stratified, `random_state=42` -> 10,888 train / 2,723 test |

**Features:** `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`, `Compactness`, `ShapeFactor1`-`ShapeFactor4`.

**Class balance (full data):** Dermason 3,546 - Sira 2,636 - Seker 2,027 - Horoz 1,928 - Cali 1,630 - Barbunya 1,322 - Bombay 522 (mildly imbalanced, so macro-averaged metrics are reported).

## c. GitHub repository link

<https://github.com/Anshitseth/dry-bean-classifier-ml>

```
dry-bean-classifier/
|-- app.py                 Streamlit app
|-- train_models.py        trains + tunes + saves all five models
|-- metrics_utils.py       metric computation shared by training and the app
|-- requirements.txt
|-- README.md
|-- test_data.csv          held-out 20 % test split (2,723 rows, 16 features + Class)
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- label_encoder.joblib
|   |-- feature_names.joblib
|   `-- metrics.csv
```

Live Streamlit app: <https://dry-bean-classifier-ml-2026aj05244.streamlit.app/>

## d. Models used

Each model is a scikit-learn `Pipeline` (a `StandardScaler` is included for Logistic Regression, KNN, Naive Bayes and SVM; the Decision Tree uses raw features). Hyper-parameters were picked with 3-fold stratified `GridSearchCV` on the training split only (scoring = accuracy). The test split was used once, for the numbers below.

Precision, Recall and F1 are **macro-averaged**; AUC is **one-vs-rest macro**; MCC is the multi-class generalisation.

### Comparison table (test set, 2,723 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9218 | 0.9948 | 0.9359 | 0.9307 | 0.9330 | 0.9055 |
| Decision Tree | 0.9093 | 0.9787 | 0.9253 | 0.9207 | 0.9228 | 0.8903 |
| KNN | 0.9188 | 0.9892 | 0.9350 | 0.9303 | 0.9323 | 0.9019 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | **0.9243** | 0.9946 | **0.9372** | **0.9350** | **0.9361** | **0.9085** |

Best hyper-parameters: LR `C=10`; DT `max_depth=12, min_samples_leaf=10`; KNN `k=15, weights=distance`; NB `var_smoothing=1e-9`; SVM `RBF, C=10, gamma=scale`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A linear model reaches 92.2 % accuracy and the highest AUC (0.9948), so the classes are largely linearly separable once features are standardised. Very fast and stable; its weak spot is Sira (F1 0.865), which overlaps with Dermason. |
| Decision Tree | Lowest AUC (0.9787) and 90.9 % accuracy. Hard splits give coarse probability estimates, hurting ranking quality, and a depth-limited tree cannot model the smooth boundaries between similar beans. Most interpretable, but the weakest of the discriminative models. |
| KNN | 91.9 % accuracy and 0.9323 F1, almost on par with Logistic Regression. Benefits from scaling and distance weighting; it perfectly isolates Bombay (F1 1.0). Slower at prediction and the largest model file, with no accuracy advantage over simpler models. |
| Naive Bayes | Weakest accuracy (89.8 %) and MCC (0.877) because the features (Area, Perimeter, ConvexArea, EquivDiameter ...) are strongly correlated, violating the independence assumption; Barbunya vs Cali confusion is highest here (F1 0.846 / 0.892). Its AUC is still high (0.9916), meaning the ranking is fine but the decision thresholds are off. |
| Support Vector Machine | Best on Accuracy (92.4 %), Precision, Recall, F1 and MCC. The RBF kernel captures the non-linear overlap between Barbunya/Cali and Dermason/Sira better than the linear model. Slower to train, but the saved model is small and predictions are fast. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF)** - top on 5 of 6 metrics (Accuracy 0.9243, F1 0.9361, MCC 0.9085) and within 0.0002 of the best AUC. Margins over Logistic Regression and KNN are small (< 0.3 pts accuracy), so on a 2,723-row test set these three are close to statistically tied; SVM is still the best overall choice. |

**Common pattern:** every model classifies Bombay almost perfectly (it is much larger than other beans), and the hardest pair is **Dermason vs Sira** (most errors for all models).

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py          # optional - regenerates model/ and test_data.csv
streamlit run app.py
```

`train_models.py` downloads the dataset through the `ucimlrepo` package (`pip install ucimlrepo`; not needed to run the app).

## Streamlit app features

- **CSV upload** of test data (must contain the 16 feature columns and `Class`); a bundled `test_data.csv` can be used instead.
- **Model dropdown** to select any of the five saved models.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix** heat-map and **classification report**.
- **Comparison of all five models** on the uploaded file with a bar chart.

Models are loaded from `model/*.joblib`; the app never retrains.

## BITS Virtual Lab

Screenshot of the assignment run on BITS Virtual Lab is included in the submitted PDF.
