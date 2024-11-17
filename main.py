import pandas as pd
import pickle

from typing import Any

from Trainable import Trainable

from __future__ import  annotations
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

#from scipy.stats import chisquare, ks_2samp

app:FastAPI = FastAPI()

model_name:str = ""

with open("./artifacts/"+ model_name, "r+") as file:
    model:Any = pickle.load(file=file)

class InputData(BaseModel):
    feature1: float #TODO : I dunno if we need this implementation
    pass 

@app.get("/health")
async def health_check():
    return {
        "status" : "All Gucci" #TODO: Change this shit
    }

@app.post("/predict")
async def predict(data: InputData):
    
    input_df:pd.DataFrame = ([]) 

    # TODO: Add something to scale the data (our own simple preprocessing, or smth like that), i dunno uwu

    try:
        prediction = model.predict(data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    #TODO: Save the prediction somewhere else, sqlite or somewhere else

    return {"prediction" : prediction}

