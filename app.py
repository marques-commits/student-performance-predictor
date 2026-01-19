<<<<<<< HEAD
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model, scaler, and label encoder
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')
le_race = joblib.load('le_race.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        gender = request.form['gender']
        parental_edu = request.form['parental_edu']
        lunch = request.form['lunch']
        test_prep = request.form['test_prep']
        math = float(request.form['math'])
        reading = float(request.form['reading'])
        writing = float(request.form['writing'])

        # Encode categorical variables (must match how they were encoded during training)
        gender_enc = 0 if gender == 'female' else 1
        parental_options = [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ]
        parental_enc = parental_options.index(parental_edu)
        lunch_enc = 1 if lunch == 'standard' else 0
        test_prep_enc = 0 if test_prep == 'none' else 1

        # Scale the three scores
        scores_scaled = scaler.transform([[math, reading, writing]])

        # Build the feature array in the same order as training data
        features = np.array([[
            gender_enc,
            parental_enc,
            lunch_enc,
            test_prep_enc,
            scores_scaled[0][0],
            scores_scaled[0][1],
            scores_scaled[0][2]
        ]])

        # Make prediction
        pred_encoded = model.predict(features)[0]
        predicted_group = le_race.inverse_transform([pred_encoded])[0]

        # Disclaimer message
        disclaimer = (
            "Important Note: The dataset uses anonymous group codes for privacy reasons. "
            "These group labels do NOT correspond to specific ethnicities such as Black, White, Latino, Asian, etc. "
            "The prediction is based solely on patterns in the provided data (scores, gender, parental education, lunch type, and test preparation)."
        )

        return render_template(
            'index.html',
            prediction=predicted_group,
            disclaimer=disclaimer
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {str(e)}. Please check your input values."
        )

if __name__ == '__main__':
=======
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model, scaler, and label encoder
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')
le_race = joblib.load('le_race.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        gender = request.form['gender']
        parental_edu = request.form['parental_edu']
        lunch = request.form['lunch']
        test_prep = request.form['test_prep']
        math = float(request.form['math'])
        reading = float(request.form['reading'])
        writing = float(request.form['writing'])

        # Encode categorical variables (must match how they were encoded during training)
        gender_enc = 0 if gender == 'female' else 1
        parental_options = [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ]
        parental_enc = parental_options.index(parental_edu)
        lunch_enc = 1 if lunch == 'standard' else 0
        test_prep_enc = 0 if test_prep == 'none' else 1

        # Scale the three scores
        scores_scaled = scaler.transform([[math, reading, writing]])

        # Build the feature array in the same order as training data
        features = np.array([[
            gender_enc,
            parental_enc,
            lunch_enc,
            test_prep_enc,
            scores_scaled[0][0],
            scores_scaled[0][1],
            scores_scaled[0][2]
        ]])

        # Make prediction
        pred_encoded = model.predict(features)[0]
        predicted_group = le_race.inverse_transform([pred_encoded])[0]

        # Disclaimer message
        disclaimer = (
            "Important Note: The dataset uses anonymous group codes for privacy reasons. "
            "These group labels do NOT correspond to specific ethnicities such as Black, White, Latino, Asian, etc. "
            "The prediction is based solely on patterns in the provided data (scores, gender, parental education, lunch type, and test preparation)."
        )

        return render_template(
            'index.html',
            prediction=predicted_group,
            disclaimer=disclaimer
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {str(e)}. Please check your input values."
        )

if __name__ == '__main__':
>>>>>>> 8eb01d4e08b0c8ac3d360fff3d6c842aa91b6418
    app.run(debug=True)