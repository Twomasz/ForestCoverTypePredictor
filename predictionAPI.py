from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from SupportScript import SimpleHeuristic

app = FastAPI()


class InputData(BaseModel):
    Elevation: int
    Aspect: int
    Slope: int
    HydroHorDist: int
    HydroVertDist: int
    RoadHorDist: int
    Hillshade_9am: int
    Hillshade_Noon: int
    Hillshade_3pm: int
    FirePtsHorDist: int
    Rawah_WA: int
    Neota_WA: int
    ComanchePeak_WA: int
    CacheLaPoudre_WA: int
    SoilType_0: int
    SoilType_1: int
    SoilType_2: int
    SoilType_3: int
    SoilType_4: int
    SoilType_5: int
    SoilType_6: int
    SoilType_7: int
    SoilType_8: int
    SoilType_9: int
    SoilType_10: int
    SoilType_11: int
    SoilType_12: int
    SoilType_13: int
    SoilType_14: int
    SoilType_15: int
    SoilType_16: int
    SoilType_17: int
    SoilType_18: int
    SoilType_19: int
    SoilType_20: int
    SoilType_21: int
    SoilType_22: int
    SoilType_23: int
    SoilType_24: int
    SoilType_25: int
    SoilType_26: int
    SoilType_27: int
    SoilType_28: int
    SoilType_29: int
    SoilType_30: int
    SoilType_31: int
    SoilType_32: int
    SoilType_33: int
    SoilType_34: int
    SoilType_35: int
    SoilType_36: int
    SoilType_37: int
    SoilType_38: int
    SoilType_39: int


@app.get("/")
async def root():
    return {"message": "Hello OpenX!"}


@app.post("/simple-heuristic-prediction")
async def simple_heuristic_prediction(data: InputData):
    sample_features = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    my_heuristic = SimpleHeuristic()
    my_heuristic.load_weights('models/SimpleHeuristicWeights.npy')

    y_pred = my_heuristic.single_predict(sample_features.squeeze())

    return {'Predicted': f'{y_pred}'}


@app.post("/nearest-neighbors-prediction")
async def nearest_neighbors_prediction(data: InputData):
    sample_features = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    model = pickle.load(open('models/NearestNeighbors.pickle', "rb"))
    ct = pickle.load(open('models/ColumnTransformer.pickle', "rb"))

    X_tran = ct.transform(sample_features)

    y_pred = model.predict(X_tran)[0]

    return {'Predicted': f'{y_pred}'}


@app.post("/random-forest-prediction")
async def random_forest_prediction(data: InputData):
    sample_features = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    model = pickle.load(open('models/RandomForest.pickle', "rb"))

    y_pred = model.predict(sample_features)[0]

    return {'Predicted': f'{y_pred}'}


@app.post("/neural-network-prediction")
async def neural_network_prediction(data: InputData):
    sample_features = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    model = pickle.load(open('models/NeuralNetwork.pickle', "rb"))
    ct = pickle.load(open('models/ColumnTransformer.pickle', "rb"))

    X_tran = ct.transform(sample_features)
    predictions = model.predict(X_tran)

    y_pred = np.argmax(predictions)

    return {'Predicted': f'{y_pred}'}
