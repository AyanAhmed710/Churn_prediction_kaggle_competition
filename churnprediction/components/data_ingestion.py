import os
from churnprediction.entity.config_entity import DataIngestionConfig        
import sys
from churnprediction.logging.logger import get_logger
from churnprediction.exception.exception import ChurnPredictionException   
from churnprediction.entity.artifact import DataIngestionArtifact 
import pymongo
import certifi
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
load_dotenv()

data_ingestion_logger = get_logger("data_ingestion")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.data_ingestion_logger = get_logger("data_ingestion")

    def mongodb_to_database(self):
        try:
            self.data_ingestion_logger.info("Starting data ingestion from MongoDB to database")
            client=pymongo.MongoClient(os.getenv("MONGODB_URI"), tlsCAFile=certifi.where())
            db = client[self.data_ingestion_config.database_name]
            collection = db[self.data_ingestion_config.collection_name]
            df=pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df.drop("_id", axis=1, inplace=True)
                
            self.data_ingestion_logger.info("Data ingestion from MongoDB to database completed successfully")

        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
        return df
    
    def df_to_feature_store(self, df: pd.DataFrame):
        try:
            self.data_ingestion_logger.info("Starting data ingestion from DataFrame to feature store")
            os.makedirs(self.data_ingestion_config.feature_store_dir, exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_file_path, index=False)
            self.data_ingestion_logger.info("Data ingestion from DataFrame to feature store completed successfully")

        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
    def train_test_split(self, df: pd.DataFrame):
        try:
            self.data_ingestion_logger.info("Starting train test split")
            os.makedirs(self.data_ingestion_config.ingested_dir, exist_ok=True)
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=self.data_ingestion_config.random_state)
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False)
            self.data_ingestion_logger.info("Train test split completed successfully")

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def initiate_data_ingestion(self):
        try:
            self.data_ingestion_logger.info("Starting data ingestion process")
            df=self.mongodb_to_database()
            self.df_to_feature_store(df)
            self.train_test_split(df)
            data_ingestion_artifact = DataIngestionArtifact(feature_store_file_path=self.data_ingestion_config.raw_file_path, train_file_path=self.data_ingestion_config.train_file_path, test_file_path=self.data_ingestion_config.test_file_path)
           
            self.data_ingestion_logger.info("Data ingestion process completed successfully")

        except Exception as e:
            raise ChurnPredictionException(e, sys)
        

        return data_ingestion_artifact