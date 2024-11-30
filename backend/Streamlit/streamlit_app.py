# Import Libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the best model
best_model = joblib.load("best_model.pkl")

# Streamlit app title
st.title("Iris Flower Prediction App")

# Description of the app
st.write(""" This app predicts the species of the Iris flower based on its physical attributes (sepal length, sepal width, petal length,
             and petal width). Simply enter the values below to get the predicted species.""")

# Create input fields for the user to enter the features
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2)

# Button to make the prediction
if st.button("Predict Species"):
    # Prepare the input data as a numpy array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make the prediction
    prediction = best_model.predict(input_data)
    
    # Display the result
    st.write(f"The predicted Iris species is: **{prediction[0]}**")

