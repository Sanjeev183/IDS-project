import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data = pd.read_csv("dataset/NSL_KDD_dataset.csv")
data = data.drop_duplicates()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
min_samples = 10
data = data.groupby("label").filter(lambda x: len(x) >= min_samples)

# Encode categorical features
protocol_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()
label_encoder = LabelEncoder()

data["protocol_type"] = protocol_encoder.fit_transform(data["protocol_type"])
data["service"] = service_encoder.fit_transform(data["service"])
data["flag"] = flag_encoder.fit_transform(data["flag"])
data["label"] = label_encoder.fit_transform(data["label"])

# Features and target
X = data.drop("label", axis=1)
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Random Forest with balanced regularization
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

# Train on full training data
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ids_model.pkl")
joblib.dump(
    {
        "protocol": protocol_encoder,
        "service": service_encoder,
        "flag": flag_encoder,
        "label": label_encoder,
    },
    "model/encoder.pkl"
)

print("\nModel and encoders saved successfully!")