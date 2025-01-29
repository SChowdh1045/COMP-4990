from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Load the trained pipeline
with open('model_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Validate numerical inputs
        if not (0 <= float(data['hoursStudied']) <= 80):  # Max ~11.4 hrs/day for 7 days
            raise ValueError("Hours studied per week must be between 0 and 80")
        
        if not (0 <= float(data['attendance']) <= 100):
            raise ValueError("Attendance percentage must be between 0 and 100")
        
        if not (0 <= float(data['previousScores']) <= 100):
            raise ValueError("Previous scores must be between 0 and 100")
        
        if not (0 <= float(data['tutoringSessions']) <= 30):  # Max 1 session per day
            raise ValueError("Monthly tutoring sessions must be between 0 and 30")
        
        if not (0 <= float(data['physicalActivity']) <= 40):
            raise ValueError("Physical activity hours must be between 0 and 40")
        

        # Creating DataFrame with raw features
        input_data = pd.DataFrame({
            # Numerical features
            'Hours_Studied': [float(data['hoursStudied'])],
            'Previous_Scores': [float(data['previousScores'])],
            'Physical_Activity': [float(data['physicalActivity'])],
            'Attendance': [float(data['attendance'])],
            'Tutoring_Sessions': [float(data['tutoringSessions'])],
            
            # Binary features
            'Extracurricular_Activities': [data['extracurricularActivities']],
            'Internet_Access': [data['internetAccess']],
            'Learning_Disabilities': [data['learningDisabilities']],
            'Gender': [data['gender']],
            
            # Multi-category features
            'Distance_from_Home': [data['distanceFromHome']],
            'Parental_Involvement': [data['parentalInvolvement']],
            'Access_to_Resources': [data['accessToResources']],
            'Motivation_Level': [data['motivationLevel']],
            'Family_Income': [data['familyIncome']],
            'Teacher_Quality': [data['teacherQuality']],
            'Peer_Influence': [data['peerInfluence']],
            'Parental_Education_Level': [data['parentalEducationLevel']]
        })
        
        # Make prediction
        prediction = pipeline.predict(input_data)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction[0]),
            'message': 'Prediction successful'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error making prediction'
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)