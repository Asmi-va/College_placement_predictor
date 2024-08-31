import numpy as np
import pickle
from flask import Flask, render_template, request


app = Flask(__name__, template_folder="templates")

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Get parameters from request
    gender = request.args.get('gender')
    stream = request.args.get('stream')
    internship = request.args.get('internship')
    cgpa = float(request.args.get('cgpa'))  # Convert to float
    backlogs = float(request.args.get('backlogs'))  # Convert to float
    
    # Convert input to numpy array
    input_data = np.array([gender, stream, internship, cgpa, backlogs])
    input_data = input_data.astype(float)  # Ensure data type is float
    
    # Make prediction
    output = model.predict([input_data])
    
    # Determine output message based on prediction
    if output == 1:
        out = 'You have high chances of getting placed!!!'
    else:
        out = 'You have low chances of getting placed. All the best.'
    
    return render_template('out.html', output=out)

if __name__ == "__main__":
    app.run(debug=True)
