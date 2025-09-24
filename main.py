from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import numpy as np

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# ========== TRAIN MODEL AT STARTUP ==========
iris = load_iris()
X, y = iris.data, iris.target
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

target_names = iris.target_names

# ========== FASTAPI APP ==========
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ========= SCHEMA FOR JSON API ==========
class Features(BaseModel):
    features: List[float]

# ========= ROUTES ==========

# HTML FORM
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# FORM PREDICTION
@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    species = target_names[prediction[0]]
    return templates.TemplateResponse("index.html", {"request": request, "result": species})

# JSON API
@app.post("/predict")
async def predict(data: Features):
    if len(data.features) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 features required.")
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    species = target_names[prediction[0]]
    return {"prediction": int(prediction[0]), "species": species}
