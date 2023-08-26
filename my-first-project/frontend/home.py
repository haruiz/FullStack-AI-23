import streamlit as st
import pandas as pd
import random
import requests
import json

url = "http://localhost:8000/predict"

df = pd.DataFrame(
    {
        "name": ["Roadmap", "Extras", "Issues"],
        "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app", "https://issues.streamlit.app"],
        "stars": [random.randint(0, 1000) for _ in range(3)],
        "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
    }
)

def call_api(sepal_length, sepal_width, petal_length, petal_width):
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
    response = requests.request("POST", url, headers=headers, data=request_data_json)
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
        label = call_api(sepal_length, sepal_width, petal_length, petal_width)
        st.write(f'Predicted label: {label}')
        st.balloons()

    # st.dataframe(
    # df,
    # column_config={
    #     "name": "App name",
    #     "stars": st.column_config.NumberColumn(
    #         "Github Stars",
    #         help="Number of stars on GitHub",
    #         format="%d ⭐",
    #     ),
    #     "url": st.column_config.LinkColumn("App URL"),
    #     "views_history": st.column_config.LineChartColumn(
    #         "Views (past 30 days)", y_min=0, y_max=5000
    #     ),
    # },
    # hide_index=True,
    # )
   
    # st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

    # with st.expander("See explanation"):
    #     st.write("""
    #         The chart above shows some numbers I picked for you.
    #         I rolled actual dice for these, so they're *guaranteed* to
    #         be random.
    #     """)
    #     st.image("https://static.streamlit.io/examples/dice.jpg")



if __name__ == '__main__':
    app()
