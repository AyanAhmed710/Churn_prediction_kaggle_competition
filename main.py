import os
import dagshub
import sys
from dotenv import load_dotenv


load_dotenv()

# Only set if not already defined in the environment (EC2 will have these pre-set)
if os.environ.get("DAGSHUB_USERNAME") and os.environ.get("DAGSHUB_TOKEN"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

dagshub.init(repo_owner='sheikhayanahmad710', repo_name='Churn_prediction_kaggle_competition', mlflow=True)
import io

# Fix Windows cp1252 encoding issue with MLflow emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pandas as pd
from churnprediction.components.data_validation import DataValidation
from churnprediction.components.data_transformation import DataTransformation
from churnprediction.components.model_trainer import ModelTrainer
from churnprediction.entity.artifact import DataIngestionArtifact ,DataValidationArtifact ,ModelTrainerArtifact
from churnprediction.components.data_ingestion import DataIngestion
from churnprediction.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig ,DataValidationConfig ,DataTransformationConfig ,ModelTrainingConfig
from churnprediction.logging.logger import get_logger
from dotenv import load_dotenv

main_logger = get_logger("main")


if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()
    # data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    # data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    # data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_ingestion_artifact = DataIngestionArtifact(feature_store_file_path=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\Artifact\20260320005439\data_ingestion\feature_store\churn.csv"   ,train_file_path=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\Artifact\20260320005439\data_ingestion\ingested\train.csv", test_file_path=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\Artifact\20260320005439\data_ingestion\ingested\test.csv")

    data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
    data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
    data_validation_artifact = data_validation.initiate_data_validation()
    main_logger.info(f"Data validation artifact: {data_validation_artifact}")
    main_logger.info("Data validation completed successfully.")

    main_logger.info("Starting data transformation process")
    data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
    data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    

    main_logger.info("Starting model trainer process")
    model_trainer_config = ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
    model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
    model_trainer_artifact = model_trainer.initialize_model_training()
    main_logger.info(f"Model trainer artifact: {model_trainer_artifact}")
    main_logger.info("Model trainer completed successfully.")