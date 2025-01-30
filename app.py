from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the pre-trained model
with open("loan_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form

        # Extract features from form data
        input_features = [
            int(data['id']),
            int(data['person_age']),
            int(data['person_income']),
            int(data['person_home_ownership']),
            float(data['person_emp_length']),
            int(data['loan_intent']),
            int(data['loan_grade']),
            int(data['loan_amnt']),
            float(data['loan_int_rate']),
            float(data['loan_percent_income']),
            int(data['cb_person_default_on_file']),
            int(data['cb_person_cred_hist_length'])
        ]

        # Convert to numpy array for prediction
        input_array = np.array(input_features).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(input_array)

        # Determine loan status
        loan_status = "Approved" if prediction[0] == 1 else "Rejected"

        return render_template('index.html', loan_status=loan_status)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
