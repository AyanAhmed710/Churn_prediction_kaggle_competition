import os
import sys

from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import get_logger
from churnprediction.entity.config_entity import TrainingPipelineConfig
from churnprediction.entity.artifact import DataIngestionArtifact ,DataValidationArtifact ,DataTransformationArtifact ,ModelTrainerArtifact
from churnprediction.components.data_ingestion import DataIngestion
from churnprediction.components.data_validation import DataValidation
from churnprediction.components.data_transformation import DataTransformation
from churnprediction.components.model_trainer import ModelTrainer
from churnprediction.cloud.sync_s3 import S3_Sync
from churnprediction.utils import save_object ,load_object
from churnprediction.constants.training_pipeline import FINAL_PUSHED_MODEL_PATH ,BUCKET_NAME ,FINAL_PUSHED_PREPROCESSOR_PATH
from churnprediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig)



training_logger=get_logger("TrainingPipeline")

class Training_Pipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        self.s3_sync=S3_Sync()

    def start_data_ingestion(self):

        try:
            training_logger.info("Starting data ingestion process")
            self.data_ingestion_config=DataIngestionConfig(TrainingPipelineConfig())
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            training_logger.info("Data ingestion completed successfully")
            return data_ingestion_artifact

        except Exception as e:
            raise ChurnPredictionException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            training_logger.info("Starting data validation process")
            data_validation_config=DataValidationConfig(TrainingPipelineConfig())
            data_validation=DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact=data_validation.initiate_data_validation()
            training_logger.info("Data validation completed successfully")
            return data_validation_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        

    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            training_logger.info("Starting data transformation process")
            data_transformation_config=DataTransformationConfig(TrainingPipelineConfig())
            data_transformation=DataTransformation(data_transformation_config=data_transformation_config,data_validation_artifact=data_validation_artifact)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            training_logger.info("Data transformation completed successfully")
            return data_transformation_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        

    def start_data_training(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            training_logger.info("Starting data training process")
            model_trainer_config=ModelTrainingConfig(TrainingPipelineConfig())
            model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact=model_trainer.initialize_model_training()
            training_logger.info("Data training completed successfully")
            push_status=True

            if push_status ==True :
                save_object(file_path=FINAL_PUSHED_MODEL_PATH, object=load_object(model_trainer_artifact.model_file_path))
                save_object(file_path=FINAL_PUSHED_PREPROCESSOR_PATH, object=load_object(data_transformation_artifact.transformed_object_file_path))
            return model_trainer_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        

    def run_pipeline(self):
        try:
            # data_ingestion_artifact=self.start_data_ingestion()
            data_ingestion_artifact = DataIngestionArtifact(feature_store_file_path=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\Artifact\20260320005439\data_ingestion\feature_store\churn.csv"   ,train_file_path=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\Artifact\20260320005439\data_ingestion\ingested\train.csv", test_file_path=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\Artifact\20260320005439\data_ingestion\ingested\test.csv")
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_data_training(data_transformation_artifact=data_transformation_artifact)
            self.syn_artifact_dir_s3()
            self.sync_saved_model_dir_s3()
            training_logger.info("Pipeline completed successfully")

            return model_trainer_artifact
        except Exception as e:
            raise ChurnPredictionException(e,sys)
        

    def syn_artifact_dir_s3(self):

        try:
            aws_url_bucket=f"s3://{BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(self.training_pipeline_config.artifact_dir, aws_url_bucket)

        except Exception as e:
            raise ChurnPredictionException(e,sys)
        

    def sync_saved_model_dir_s3(self):

        try:
            aws_url_bucket=f"s3://{BUCKET_NAME}/saved_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(self.training_pipeline_config.model_dir, aws_url_bucket)

        except Exception as e:
            raise ChurnPredictionException(e,sys)


    