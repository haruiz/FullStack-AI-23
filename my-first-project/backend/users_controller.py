from pydantic import BaseModel
import numpy as np
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

class User(BaseModel):
    name: str
    email: str
    password: str

router = InferringRouter()


@cbv(router)
class UsersController:
    @router.get("/hi")
    def hi(self):
        return "hi from UsersController"