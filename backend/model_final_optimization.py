import pandas as pd
from scipy.io import arff

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


print("Loading dataset...")

# ==========================================================
# LOAD DATASET
# ==========================================================

data, meta = arff.loadarff("phishing.arff")
df = pd.DataFrame(data)

# Decode byte strings
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode("utf-8")

# Convert everything to numeric (FIX FOR XGBOOST ERROR)
df = df.apply(pd.to_numeric)

# Split features and label
X = df.drop(columns=["Result"])
y = df["Result"]

# ==========================================================
# CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# ==========================================================
# OPTIMIZED EXTRATREES
# ==========================================================

print("\nRunning Optimized ExtraTrees...")

et = ExtraTreesClassifier(
    n_estimators=900,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

scores = cross_val_score(
    et,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("ExtraTrees Accuracy:", scores.mean())

# ==========================================================
# STACKING ENSEMBLE
# ==========================================================

print("\nRunning Advanced Stacking Ensemble...")

rf = RandomForestClassifier(
    n_estimators=800,
    n_jobs=-1
)

xgb = XGBClassifier(
    eval_metric="logloss",
    n_estimators=500,
    n_jobs=-1
)

lgb = LGBMClassifier(
    n_estimators=500
)

svm = SVC(
    probability=True
)

stack = StackingClassifier(

    estimators=[
        ("rf", rf),
        ("et", et),
        ("xgb", xgb),
        ("lgb", lgb),
        ("svm", svm)
    ],

    final_estimator=LogisticRegression(),

    cv=5,

    n_jobs=-1
)

scores = cross_val_score(
    stack,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("Stacking Accuracy:", scores.mean())