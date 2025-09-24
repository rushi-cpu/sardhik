from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
from sklearn.datasets import load_iris

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

iris = load_iris()
target_names = iris.target_names

# Initialize app
app = FastAPI()

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# HTML Home Page
@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Form submission endpoint
@app.post("/predict_web", response_class=HTMLResponse)
def predict_web(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    prediction = model.predict(features)
    species = target_names[prediction[0]]
    return templates.TemplateResponse("index.html", {"request": request, "result": species})

# JSON API (same as before)
class Features(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(data: Features):
    if len(data.features) != 4:
        raise HTTPException(status_code=400, detail="Input must be 4 features")

    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    species = target_names[prediction[0]]

    return {
        "prediction": int(prediction[0]),
        "species": species
    }
