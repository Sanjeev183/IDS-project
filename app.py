from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model/ids_model.pkl")

# Load encoders
encoders = joblib.load("model/encoder.pkl")

protocol_encoder = encoders["protocol"]
service_encoder = encoders["service"]
flag_encoder = encoders["flag"]
label_encoder = encoders["label"]


# HOME PAGE
@app.route("/")
def index():
    return render_template("index.html")


# DETECT PAGE 
@app.route("/detect")
def detect():
    return render_template("detect.html")


# ABOUT PAGE 
@app.route("/about")
def about():
    return render_template("about.html")


# PREDICTION 
@app.route("/predict", methods=["POST"])
def predict():

    try:
        duration = int(request.form["duration"])
        protocol = request.form["protocol"]
        service = request.form["service"]
        flag = request.form["flag"]
        src_bytes = int(request.form["src_bytes"])
        dst_bytes = int(request.form["dst_bytes"])
        land = int(request.form["land"])
        wrong_fragment = int(request.form["wrong_fragment"])
        urgent = int(request.form["urgent"])

        # Encode categorical features
        protocol = protocol_encoder.transform([protocol])[0]
        service = service_encoder.transform([service])[0]
        flag = flag_encoder.transform([flag])[0]

        # Create dataframe
        input_df = pd.DataFrame([[
            duration,
            protocol,
            service,
            flag,
            src_bytes,
            dst_bytes,
            land,
            wrong_fragment,
            urgent
        ]], columns=[
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent"
        ])

        # Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        confidence = max(probability[0]) * 100

        result = label_encoder.inverse_transform(prediction)[0]

        final_result = f"{result.upper()} (Confidence: {confidence:.2f}%)"

    except Exception as e:
        final_result = "Error: " + str(e)

    return render_template("result.html", prediction_text=final_result)


# RUN SERVER 
if __name__ == "__main__":
    app.run(debug=True)