from flask import Flask, render_template, request
import joblib
import traceback
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "model.pkl"
model = joblib.load(model_path)

# Mapping for categorical columns
location_mapping = {
    "Rajwada": 0, "Palasia": 1, "AB Road": 2,
    "Vijay Nagar": 3, "Geeta Bhavan": 4,
    "Rau": 5, "Bhawarkua": 6,"Mahalaxmi Nagar": 7
}
furnished_mapping = {
    "Furnished": 0, "Semi-Furnished": 1, "Unfurnished": 2
}

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        location = request.form.get('location', None)
        furnished = request.form.get('furnished', None)
        bedrooms = request.form.get('bedrooms', None)

        # Debug input values
        print(f"Location: {location}, Furnished: {furnished}, Bedrooms: {bedrooms}")

        # Validate inputs
        if not location or not furnished or not bedrooms:
            return render_template('index.html', prediction_text="Please provide all inputs.")

        if location not in location_mapping or furnished not in furnished_mapping:
            return render_template('index.html', prediction_text="Invalid location or furnishing type.")

        # Encode inputs
        location_encoded = location_mapping[location]
        furnished_encoded = furnished_mapping[furnished]
        bedrooms = int(bedrooms)

        # Prepare input for model
        # Match the feature order used during training
        features = np.array([[bedrooms, location_encoded, furnished_encoded]])

        # Predict house price
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f"Predicted Price: ₹{prediction:.2f} Lakhs")

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return render_template('index.html', prediction_text="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
