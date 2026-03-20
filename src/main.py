
# Intrusion Detection System (IDS)

# STEP 1: Import required libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# STEP 2: Load the dataset
data = pd.read_csv("dataset/NSL_KDD_dataset.csv")


# STEP 3: Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]


# STEP 4: Encode target labels (normal -> 0 | attack -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# STEP 5: Convert categorical input features to numeric
X = pd.get_dummies(X)


# STEP 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 7: Create and train the ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# STEP 8: Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# STEP 9: Save the trained model
joblib.dump(model, "model/ids_model.pkl")