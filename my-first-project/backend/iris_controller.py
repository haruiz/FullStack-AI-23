from pydantic import BaseModel
import numpy as np
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
import typing
from fastapi import Depends, Request
from model_loader import ModelLoader

class IrisModelRow(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    def to_numpy(self):
        return np.array([
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width
        ])
    

router = InferringRouter()

async def get_model(request: Request):
    return request.app.state.model


@cbv(router)
class IrisController:

    model: ModelLoader = Depends(get_model)

    @router.get("/hi")
    def hi(self):
        return "hi from iris controller running in docker, with reloading!"
    
    @router.post("/predict")
    def predict(self, rows: typing.List[IrisModelRow]):  
        rows = np.array([row.to_numpy() for row in rows])
        predictions = self.model.predict(rows)
        return {"predictions": predictions}
    
