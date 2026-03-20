import joblib
import pandas as pd


# Load model and encoders once

model = joblib.load("model/ids_model.pkl")
encoders = joblib.load("model/encoder.pkl")

protocol_encoder = encoders["protocol"]
service_encoder = encoders["service"]
flag_encoder = encoders["flag"]
label_encoder = encoders["label"]


# Prediction function

def predict_intrusion(input_data):
    """
    input_data: dict with keys -
        duration, protocol_type, service, flag,
        src_bytes, dst_bytes, land, wrong_fragment, urgent
    Returns: string with predicted label and confidence
    """

    columns = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent"
    ]

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data], columns=columns)


    # Encode categorical features
    
    try:
        input_df["protocol_type"] = protocol_encoder.transform(input_df["protocol_type"])
        input_df["service"] = service_encoder.transform(input_df["service"])
        input_df["flag"] = flag_encoder.transform(input_df["flag"])
    except ValueError as e:
        # This happens if a category was not seen during training
        return f"Error: Invalid categorical input! ({e})"

  
    # Make prediction
  
    result = model.predict(input_df)

    # Predict probability for confidence
    try:
        prob = model.predict_proba(input_df)
        confidence = float(max(prob[0]) * 100)
    except AttributeError:
        # If model doesn't support predict_proba (just in case)
        confidence = None

    # Decode predicted label
    prediction = label_encoder.inverse_transform(result)

    if confidence is not None:
        return f"{prediction[0].upper()} (Confidence: {confidence:.2f}%)"
    else:
        return f"{prediction[0].upper()} (Confidence not available)"