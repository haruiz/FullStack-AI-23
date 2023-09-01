import streamlit as st
import pandas as pd
import random
import requests
import json
import os

API_ENDPOINT = os.environ.get("API_ENDPOINT", "https://localhost:8080")

def call_api_predict_method(sepal_length, sepal_width, petal_length, petal_width):
    request_data = [{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }]
    request_data_json = json.dumps(request_data)
    headers = {
    'Content-Type': 'application/json'
    }
    predict_method_endpoint = f"{API_ENDPOINT}/iris/predict"
    response = requests.request("POST",predict_method_endpoint , headers=headers, data=request_data_json)
    response_json = response.json()
    predictions = response_json['predictions']
    label = predictions[0]
    return label


def app():

    st.set_page_config(
        page_title="Streamlit App",
        page_icon="🧊",
        layout="wide"
    )
    st.title('Home')
    st.write('Welcome to the Iris frontend demo!!!')

    sepal_length = st.number_input('Sepal length', min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input('Sepal width', min_value=0.0, max_value=10.0, value=5.0)
    petal_length = st.number_input('Petal length', min_value=0.0, max_value=10.0, value=5.0)
    petal_width = st.number_input('Petal width', min_value=0.0, max_value=10.0, value=5.0)

    i_was_clicked = st.button("Predict")
    if i_was_clicked:
        label = call_api_predict_method(sepal_length, sepal_width, petal_length, petal_width)
        st.write(f'Predicted label: {label}')
        st.balloons()

if __name__ == '__main__':
    app()
