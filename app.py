import pandas as pd
import streamlit as st
import pickle

# Load the pickled model
with open('Sklearn_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title("Iris Flower Classification")

# Input fields for user
sepal_length = st.slider("Sepal Length (cm):", 4.3, 7.9, 5.4)
sepal_width = st.slider("Sepal Width (cm):", 2.0, 4.4, 3.4)
petal_length = st.slider("Petal Length (cm):", 1.0, 6.9, 1.3)
petal_width = st.slider("Petal Width (cm):", 0.1, 2.5, 0.2)

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    "sepal length (cm)": [sepal_length],
    "sepal width (cm)": [sepal_width],
    "petal length (cm)": [petal_length],
    "petal width (cm)": [petal_width]
})

class_names = ['Setosa', 'Versicolor', 'Virginica']

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data)
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    result = class_names[prediction[0]]
    st.write(f"Predicted Iris Class: {result}")

# Optionally, display the prediction probability for each class
if st.checkbox("Show Class Probabilities"):
    probabilities = model.predict_proba(input_data)
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name} Probability: {probabilities[0, i]:.2f}")
