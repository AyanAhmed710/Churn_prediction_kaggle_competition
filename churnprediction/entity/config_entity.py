import os
import sys
import pandas as pd
from datetime import datetime
from churnprediction.constants import training_pipeline

class TrainingPipelineConfig:
    def __init__(self ,timestamp=datetime.now()):

            self.timestamp = timestamp.strftime("%Y%m%d%H%M%S")
            self.pipeline_name = training_pipeline.PIPELINE_NAME
            self.artifact_dir = os.path.join(training_pipeline.ARTIFACT_NAME, self.timestamp)
            self.pipeline_name = training_pipeline.PIPELINE_NAME
            self.timestamp = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_dir = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)
        self.ingested_dir = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR)
        self.train_file_path = os.path.join(self.ingested_dir, training_pipeline.TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.ingested_dir, training_pipeline.TEST_FILE_NAME)
        self.raw_file_path = os.path.join(self.feature_store_dir, training_pipeline.FILE_NAME)
        self.train_test_split_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.random_state = training_pipeline.DATA_INGESTION_RANDOM_STATE


class DataValidationConfig:
     def __init__(self, training_pipeline_config: TrainingPipelineConfig):
          
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.validated_dir = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALIDATED_DIR)
        self.invalid_dir = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path=os.path.join(self.validated_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path=os.path.join(self.validated_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path=os.path.join(self.invalid_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path=os.path.join(self.invalid_dir, training_pipeline.TEST_FILE_NAME)
        self.drift_report_dir = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR)
        self.drift_report_file_path = os.path.join(self.drift_report_dir, training_pipeline.DATA_VALIDATION_DRIFT_FILE_NAME)


class DataTransformationConfig:
     
     def __init__(self, training_pipeline_config : TrainingPipelineConfig):
          
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.data_transformation_transformed_dir = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR)
        self.transformed_train_file_path = os.path.join(self.data_transformation_transformed_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME)
        self.transformed_test_file_path = os.path.join(self.data_transformation_transformed_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME)
        self.transformed_object_file_path = os.path.join(self.data_transformation_transformed_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_NAME)
