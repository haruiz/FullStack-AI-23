from fastapi import FastAPI
import numpy as np
from model_loader import ModelLoader, Framework
from fastapi.middleware.cors import CORSMiddleware
from users_controller import router as users_router
from iris_controller import router as iris_router


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


@app.on_event("shutdown")
def shutdown_event():
    """this function will run once when the application shuts down"""
    print("Shutting down the application...")
  

app.include_router(
    users_router, 
    tags=["users"], 
    prefix="/users"
)

app.include_router(
    iris_router, 
    tags=["iris"],
    prefix="/iris"
)

@app.get("/hi")
def hi():
    return {"message": "Hello World from the API!!!"}



            
            