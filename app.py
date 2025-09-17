# Import Flask modules for creating the web app, handling templates and form data
from flask import Flask, render_template, request

# Import pickle for loading the saved scaler
import pickle

# Import numpy for handling numerical arrays
import numpy as np

# Import Keras function to load the saved deep learning model
from tensorflow.keras.models import load_model



# Initialize the Flask app
app = Flask(__name__)


# Load the trained model from the saved HDF5 file
model = load_model('models/model.h5')

# Load the pre-fitted scaler from the saved pickle file
with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Function to make predictions using the model
def make_prediction(input_data):
    # Scale the input data using the pre-fitted scaler
    input_data_scaled = scaler.transform(input_data)  # transform ensures consistent scaling

    # Predict probabilities for the scaled input
    predictions = model.predict(input_data_scaled)

    # Convert probabilities to binary class labels (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int)

    # Return the binary prediction
    return predicted_classes


# Define the route for the home page
# When user visits '/', the index.html template will be rendered
@app.route('/')
def index():
    return render_template('index.html')


# Define the route for making predictions
# This route handles both POST (form submission) and GET (visiting page) requests
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Retrieve form data from the user input
        VWTI = float(request.form['VWTI'])
        SWTI = float(request.form['SWTI'])
        CWTI = float(request.form['CWTI'])
        EI = float(request.form['EI'])

        # Arrange the input values into a 2D NumPy array for prediction
        input_data = np.array([[VWTI, SWTI, CWTI, EI]])

        # Call the make_prediction function to get the predicted class
        result = make_prediction(input_data)
        print(result)  # For debugging: prints [0] or [1]

        # Convert the numeric prediction to a human-readable string
        if result[0] == 1:
            output = "real"
        else:
            output = "fake"
        print(output)  # For debugging: prints "real" or "fake"

        # Pass the prediction to the template to display on the web page
        return render_template('index.html', prediction=output)

    # If the request method is GET (visiting the page without submitting form)
    return render_template('index.html', prediction=None)


# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
