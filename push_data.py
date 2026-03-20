import os
import pymongo
from dotenv import load_dotenv
import pandas as pd
import sys
import certifi
from churnprediction.logging.logger import get_logger
from churnprediction.exception.exception import ChurnPredictionException    

data_pusher_logger=get_logger("data_pusher")
load_dotenv()

DATABASE_NAME="churn_prediction"
COLLECTION_NAME="customer_churn_data"  
DATA_PATH=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\churn_data\churn.csv"


class MongoDBClient:
    def __init__(self):

        try:
            self.client = pymongo.MongoClient(os.getenv("MONGODB_URI"), tlsCAFile=certifi.where())
            data_pusher_logger.info("MongoDB connection established successfully")

        except Exception as e:
            ChurnPredictionException(e, sys)
        

        
    def mongodb_connection(self , database_name, collection_name):
        try:
            self.database = self.client[database_name]
            self.collection = self.database[collection_name]
            data_pusher_logger.info(f"Connected to MongoDB database: {database_name} and collection: {collection_name}")

        except Exception as e:
            ChurnPredictionException(e, sys)

        return self.collection

    def insert_data(self, data_path  ,colletion):
        try :
            data=pd.read_csv(data_path)
            data_dict=data.to_dict(orient='records')
            colletion.insert_many(data_dict)
            data_pusher_logger.info(f"Data inserted successfully from {data_path} to MongoDB collection")

        except Exception as e:
            ChurnPredictionException(e, sys)    

        return "Data inserted successfully"


if __name__=="__main__":
    mongodb_client=MongoDBClient()
    collection=mongodb_client.mongodb_connection(DATABASE_NAME, COLLECTION_NAME)
    result=mongodb_client.insert_data(DATA_PATH, collection)
    print(result)

        

