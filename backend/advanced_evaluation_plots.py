import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

# ===============================
# LOAD DATASET
# ===============================

data, meta = arff.loadarff("phishing.arff")
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode("utf-8")

X = df.drop(columns=["Result"])
y = df["Result"].apply(lambda x: 1 if str(x) == "1" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ===============================
# DEFINE BEST MODELS
# ===============================

models = {
    "RandomForest": RandomForestClassifier(n_estimators=400),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=400),
    "SVM": SVC(kernel="rbf", probability=True)
}

# ===============================
# ROC CURVE
# ===============================

plt.figure(figsize=(8,6))

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, probs)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig("roc_curve_comparison.png",dpi=300)

plt.show()

# ===============================
# PRECISION-RECALL CURVE
# ===============================

plt.figure(figsize=(8,6))

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:,1]

    precision, recall, _ = precision_recall_curve(y_test, probs)

    plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig("precision_recall_curve.png",dpi=300)

plt.show()

print("ROC and Precision-Recall graphs generated.")