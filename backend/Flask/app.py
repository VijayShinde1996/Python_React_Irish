# Import Libraries
from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_model.pkl')

# Dictionary to map species to image paths
species_to_image = {
    "setosa": "static/images/setosa.jpg",
    "versicolor": "static/images/versicolor.jpg",
    "virginica": "static/images/virginica.jpg"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    flower_image = ""
    if request.method == "POST":
        # Get user input
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Prepare the input data for the model
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make the prediction
        predicted_species = model.predict(input_data)[0]

        # Get the corresponding image
        flower_image = species_to_image.get(predicted_species, "")

        prediction = f"The predicted species is: {predicted_species.capitalize()}"

    return render_template("index.html", prediction=prediction, flower_image=flower_image)


if __name__ == "__main__":
    app.run(debug=True)
