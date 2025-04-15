
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os


app = Flask(__name__)

# Path to the model
MODEL_PATH = 'experiment_time_predictor.pkl'

# Load the model if it exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model is not None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not found. Please upload the model file.', 'success': False})

    try:
        # Get data from form
        materials = int(request.form['materials'])
        complexity = int(request.form['complexity'])
        steps = int(request.form['steps'])
        grade = int(request.form['grade'])
        subject = request.form['subject']

        # Create a DataFrame for prediction
        new_exp = pd.DataFrame({
            'Materials_Used': [materials],
            'Complexity': [complexity],
            'Steps_Number': [steps],
            'Grade': [grade],
            'Subject': [subject]
        })

        # Make prediction
        prediction = model.predict(new_exp)[0]

        return jsonify({
            'success': True,
            'prediction': round(prediction, 1),
            'input_data': {
                'materials': materials,
                'complexity': complexity,
                'steps': steps,
                'grade': grade,
                'subject': subject
            }
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global model

    if 'model_file' not in request.files:
        return jsonify({'error': 'No file part', 'success': False})

    file = request.files['model_file']

    if file.filename == '':
        return jsonify({'error': 'No selected file', 'success': False})

    try:
        file.save(MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        return jsonify({'success': True, 'message': 'Model uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)