from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import os
import joblib

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
rf_model = joblib.load("models/rf_model.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
seq_model = joblib.load("models/rf_seq_model_binary.pkl")

# Store latest result
latest_result = None
latest_next_label = None

# Main route for uploading and predicting a CSV file
@app.route("/", methods=["GET", "POST"])
def index():
    global latest_result, latest_next_label

    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded"

        # Load and clean data
        df = pd.read_csv(file)
        df = df.drop(columns=['protocol', 'service', 'state', 'attack_type'], errors='ignore')

        # Preprocess input (scale + PCA)
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)

        # Predict attack type
        y_pred = rf_model.predict(X_pca)
        labels = label_encoder.inverse_transform(y_pred)
        df["Prediction"] = labels

        # Add duration if missing
        if "duration" not in df.columns:
            df["duration"] = np.random.uniform(0.1, 10.0, size=len(df)).round(1)

        # Binary label: 0 (Normal), 1 (Malicious)
        normal_index = label_encoder.transform(["Normal"])[0]
        binary_labels = np.where(y_pred == normal_index, 0, 1)

        # Predict next step if enough data
        if len(binary_labels) >= 5:
            seq_input = binary_labels[-5:].reshape(1, -1)
            next_binary = seq_model.predict(seq_input)[0]
            latest_next_label = "Malicious" if next_binary == 1 else "Normal"
        else:
            latest_next_label = "N/A (Too few records)"
        # Save result and redirect to results page
        df.to_csv("predicted_results.csv", index=False)
        latest_result = df
        return redirect(url_for("results"))

    return render_template("index.html")

# Display prediction results summary and table
@app.route("/results")
def results():
    global latest_result, latest_next_label

    if latest_result is not None:
        df = latest_result.copy()

        # Summary counts
        total = len(df)
        malicious = (df["Prediction"] != "Normal").sum()
        benign = (df["Prediction"] == "Normal").sum()

        return render_template("result.html",
                               total=total,
                               malicious=malicious,
                               benign=benign,
                               attacks=df.to_dict(orient='records'),
                               next_label=latest_next_label)
    else:
        return redirect(url_for("index"))

# Download results
@app.route("/download")
def download():
    return send_file("predicted_results.csv", as_attachment=True)

# Start the Flask development server
if __name__ == "__main__":
    app.run(debug=True)
