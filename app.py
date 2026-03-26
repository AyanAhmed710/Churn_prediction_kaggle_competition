import sys
import os
import pymongo
import io
import json
 
from churnprediction.logging.logger import get_logger
from churnprediction.utils import load_object
 
import certifi
ca = certifi.where()
 
from dotenv import load_dotenv
load_dotenv()
 
from churnprediction.pipeline.train_pipeline import Training_Pipeline
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
 
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, JSONResponse, FileResponse
from starlette.responses import RedirectResponse
from jinja2 import Environment, FileSystemLoader
from starlette.templating import Jinja2Templates
import pandas as pd
 
from churnprediction.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME
from churnprediction.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
 
# ── Base directory ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
# ── Templates ───────────────────────────────────────────────────────────────

templates_directory=os.path.join(BASE_DIR, "templates")
    
templates = Jinja2Templates(directory=templates_directory)
 
# ── MongoDB ──────────────────────────────────────────────────────────────────
client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"))
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]
 
# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
pipeline = Training_Pipeline()
 
# ── Column order (must match training) ───────────────────────────────────────
COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]
 
 
# ── Home page ─────────────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
 
 
# ── Train ─────────────────────────────────────────────────────────────────────
@app.get("/train")
async def train():
    try:
        pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error during training: {e}")
 
 
# ── Single prediction page (GET) ──────────────────────────────────────────────
@app.get("/predict")
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})
 
 
# ── Single prediction (POST) ──────────────────────────────────────────────────
@app.post("/predict-single")
async def predict_single(request: Request):
    try:
        data = await request.json()
 
        # Build dataframe with correct column order
        df = pd.DataFrame([data])
        df = df[COLUMNS]  #
 
        # Cast numeric columns
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
        df["tenure"] = df["tenure"].astype(float)
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
        df["TotalCharges"] = df["TotalCharges"].astype(float)
 
        model_path = os.path.join(BASE_DIR, "Final_Model", "model.pkl")
        model = load_object(model_path)
 
        y_pred = model.predict_new(df)
        prediction = int(y_pred[0])
 
        return JSONResponse({"prediction": prediction})
 
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
 
 
# ── Batch prediction page (GET) ───────────────────────────────────────────────
@app.get("/predict-batch")
async def batch_page(request: Request):
    return templates.TemplateResponse("predict_batch.html", {"request": request})
 
 
# ── Batch prediction (POST) ───────────────────────────────────────────────────
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        df.drop(columns=["id"],inplace=True)
 
        model_path = os.path.join(BASE_DIR, "final_Model", "model.pkl")
        model = load_object(model_path)
 
        y_pred = model.predict_new(df)
        df["predicted_churn"] = y_pred
 
        # Save output CSV
        output_dir = os.path.join(BASE_DIR, "prediction_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output.csv")
        df.to_csv(output_path, index=False)
 
        # Return first 100 rows as JSON for table display
        preview = df.head(100).to_dict(orient="records")
        return JSONResponse({"results": preview})
 
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
 
 
# ── Download output CSV ───────────────────────────────────────────────────────
@app.get("/download-output")
async def download_output():
    output_path = os.path.join(BASE_DIR, "prediction_output", "output.csv")
    if os.path.exists(output_path):
        return FileResponse(
            output_path,
            media_type="text/csv",
            filename="churn_predictions.csv"
        )
    return Response("No predictions found. Please run batch predict first.", status_code=404)
 
 
if __name__ == "__main__":
    app_run("app:app", host="0.0.0.0", port=8000)