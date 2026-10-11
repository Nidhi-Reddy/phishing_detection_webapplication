import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

print("Loading dataset...")

data, meta = arff.loadarff("phishing.arff")
df = pd.DataFrame(data)

# Convert bytes
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode("utf-8")

# Keep ONLY 8 lexical features
selected_features = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State"
]

X = df[selected_features]
y = df["Result"].apply(lambda x: 1 if str(x) == "1" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "deployment_model.pkl")

print("Deployment model saved successfully!")