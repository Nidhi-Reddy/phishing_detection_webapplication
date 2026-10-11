# ==========================================================
# PHISHING DETECTION RESEARCH ENGINE
# FOUR DATASET CONFIGURATIONS
# ==========================================================

import pandas as pd
from scipy.io import arff
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


print("Loading phishing dataset...")

# ==========================================================
# LOAD DATASET
# ==========================================================

data, meta = arff.loadarff("phishing.arff")
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode("utf-8")

X = df.drop(columns=["Result"])
y = df["Result"].apply(lambda x: 1 if str(x) == "1" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# ==========================================================
# MODELS
# ==========================================================

models = {

"SVM": SVC(probability=True),

"RandomForest": RandomForestClassifier(n_estimators=500),

"ExtraTrees": ExtraTreesClassifier(n_estimators=500),

"GradientBoosting": GradientBoostingClassifier(),

"XGBoost": XGBClassifier(eval_metric="logloss"),

"LightGBM": LGBMClassifier(),

"MLP": MLPClassifier(max_iter=500)

}

# ==========================================================
# OPTIMIZATION PLACEHOLDERS
# ==========================================================

def grid(pipe):
    pipe.fit(X_train, y_train)
    return pipe

def random(pipe):
    pipe.fit(X_train, y_train)
    return pipe

def bayesian(pipe):
    pipe.fit(X_train, y_train)
    return pipe

def genetic(pipe):
    pipe.fit(X_train, y_train)
    return pipe

def pso(pipe):
    pipe.fit(X_train, y_train)
    return pipe

optimizations = {
"GridSearch": grid,
"RandomSearch": random,
"Bayesian": bayesian,
"Genetic": genetic,
"PSO": pso
}

# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(pipe):

    pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    return acc, prec, rec, f1

# ==========================================================
# PIPELINE BUILDER
# ==========================================================

def build_pipeline(model, use_selection=False, use_pca=False):

    steps = [("scaler", StandardScaler())]

    if use_selection:
        steps.append(("feature_selection", SelectKBest(f_classif, k=20)))

    if use_pca:
        steps.append(("pca", PCA(n_components=0.95)))

    steps.append(("clf", model))

    return Pipeline(steps)

# ==========================================================
# EXPERIMENT FUNCTION
# ==========================================================

def run_experiment(title, use_selection=False, use_pca=False):

    print("\n==============================================")
    print("DATASET:", title)
    print("==============================================")

    results = []

    for model_name, model in models.items():

        for opt_name, opt_func in optimizations.items():

            pipe = build_pipeline(model, use_selection, use_pca)

            pipe = opt_func(pipe)

            acc, prec, rec, f1 = evaluate(pipe)

            results.append([model_name, opt_name, acc, prec, rec, f1])

    results_df = pd.DataFrame(
        results,
        columns=["Model","Optimization","Accuracy","Precision","Recall","F1"]
    )

    results_df = results_df.sort_values(by="F1", ascending=False)

    print(results_df)

    return results_df

# ==========================================================
# RUN FOUR EXPERIMENTS
# ==========================================================

res1 = run_experiment("WITHOUT_SELECTION", False, False)

res2 = run_experiment("WITH_SELECTION", True, False)

res3 = run_experiment("WITHOUT_SELECTION_PCA", False, True)

res4 = run_experiment("WITH_SELECTION_PCA", True, True)

# ==========================================================
# STACKING MODEL
# ==========================================================

print("\nRunning Stacking Ensemble...")

stack = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=500)),
        ("et", ExtraTreesClassifier(n_estimators=500)),
        ("xgb", XGBClassifier(eval_metric="logloss")),
        ("lgbm", LGBMClassifier())
    ],
    final_estimator=LogisticRegression()
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", stack)
])

pipe.fit(X_train, y_train)

acc, prec, rec, f1 = evaluate(pipe)

print("\nSTACKING RESULT")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(pipe,"optimized_model.pkl")
joblib.dump(X.columns.tolist(),"feature_columns.pkl")

print("\nModel saved successfully.")