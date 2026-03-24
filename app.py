import sys
import os
import pymongo

from churnprediction.logging.logger import get_logger

import certifi
ca =certifi.where()

from dotenv import load_dotenv
load_dotenv()

from churnprediction.pipeline.train_pipeline import Training_Pipeline
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run

from fastapi import FastAPI ,File ,UploadFile ,Request
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from churnprediction.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

from churnprediction.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME

client =pymongo.MongoClient(os.getenv("MONGO_DB_URL"))
database =client[DATA_INGESTION_DATABASE_NAME]
collection=database[DATA_INGESTION_COLLECTION_NAME]

app=FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


mongo_db_url=os.getenv("MONGO_DB_URL")

pipeline=Training_Pipeline()



@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get('/train')
async def train():
    try:
        pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

if __name__ == "__main__":
    app_run("app:app", host="localhost", port=8000)