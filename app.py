from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model, label encoder, scaler, and selected features
model = joblib.load("network_attack_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("selected_features.pkl")

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = [float(request.form[col]) for col in feature_names]
        
        # Convert input to NumPy array and apply scaling
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Make prediction and get confidence score
        prediction = model.predict(input_scaled)
        prediction_prob = model.predict_proba(input_scaled)
        confidence = np.max(prediction_prob)  # Highest probability score

        # Convert numerical prediction to attack type
        attack_type = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction=attack_type, confidence=confidence, feature_names=feature_names)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
