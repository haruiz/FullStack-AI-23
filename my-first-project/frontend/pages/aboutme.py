import streamlit as st


def app():
    st.set_page_config(
        page_title="Streamlit App",
        page_icon="🧊",
        layout="wide"
    )

    st.title('About Me')
    st.write('Welcome to the Iris frontend demo!!!')
    st.write('This is a demo of a Streamlit app that can be deployed to Heroku.')
    st.write('The app is based on the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).')


if __name__ == '__main__':
    app()         
        