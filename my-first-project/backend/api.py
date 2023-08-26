from fastapi import FastAPI, Request
import numpy as np
from model_loader import ModelLoader, Framework
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.on_event("startup")
def load_model():
    """this function will run once when the application starts up"""
    print("Loading the model...")

    model = ModelLoader(
        path='models/tf/iris_model',
        framework=Framework.tensorflow,
        labels=['setosa', 'versicolor', 'virginica'],
        name='iris_model',
        version=1.0
    )
    print("Model loaded successfully!")
    app.state.model = model
  

@app.get("/")
def home():
    return {"message": "Hello World from the API"}


@app.post("/predict")
async def predict(request: Request):

    request_data_list = await request.json()
    request_data = np.array([
        list(X.values()) for X in request_data_list
    ])
    model = app.state.model
    predictions = model.predict(request_data)

    return {"predictions": predictions}




            
            