import os
import sys
import pandas as pd
from churnprediction.entity.config_entity import DataValidationConfig
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import get_logger
from churnprediction.constants import training_pipeline
from churnprediction.utils import read_yaml_file ,write_yaml_file
from scipy.stats import ks_2samp
from churnprediction.entity.artifact import DataIngestionArtifact, DataValidationArtifact



data_validation_logger = get_logger("data_validation")

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
         self.data_validation_config = data_validation_config
         self.data_ingestion_artifact = data_ingestion_artifact
         self.data_schema = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)
         data_validation_logger.info("Data validation component initialized successfully")
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    @staticmethod
    def read_data(self ,file_path: str) -> pd.DataFrame:
        try:
            data_validation_logger.info(f"Reading data from {file_path}")
            data = pd.read_csv(file_path)
            data_validation_logger.info(f"Data read successfully from {file_path}")
            return data
        except Exception as e:
            raise ChurnPredictionException(e, sys)
        

    def validate_schema(self ,df) -> bool:
        try:
            expected_columns = self.data_schema["columns"]
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                data_validation_logger.error(f"Missing columns in the ingested data: {missing_columns}")
                return False
            data_validation_logger.info("Data schema validation successful")
            return True
        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
    def check_data_types(self, df) -> bool:

     try:
        expected_data_types = self.data_schema["columns"]

        for column, expected_type in expected_data_types.items():
            if column in df.columns:
                actual_type = df[column].dtype

                if str(actual_type) != expected_type:
                    data_validation_logger.error(
                        f"Data type mismatch for column '{column}': expected {expected_type}, got {actual_type}"
                    )
                    return False

        data_validation_logger.info("Data type validation successful")
        return True

     except Exception as e:
        raise ChurnPredictionException(e, sys)
        

    def check_data_drift(self, base_df, current_df , threshold = 0.05) -> bool:
        try:
            base_columns = base_df.columns
            status=True
            report = {}

            for column in base_columns:
                base_column =base_df[column]
                current_column = current_df[column]

                p_value=float(ks_2samp(base_column, current_column).pvalue)

                if p_value < threshold:
                    data_validation_logger.warning(f"Data drift detected in column '{column}' with p-value: {p_value}")
                    is_found=True
                    status=False
                else:
                    data_validation_logger.info(f"No data drift detected in column '{column}' with p-value: {p_value}")
                    is_found=False

                report.update({column:{"p_value": p_value, "drift_status": is_found}})

            os.makedirs(self.data_validation_config.drift_report_dir, exist_ok=True)

            write_yaml_file(self.data_validation_config.drift_report_file_path, report)

            data_validation_logger.info("Data drift check completed successfully")
            return True
        except Exception as e:
            raise ChurnPredictionException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            data_validation_logger.info("Starting data validation process")
            # Check if the ingested data file exists
            train_df = DataValidation.read_data(self, self.data_ingestion_artifact.train_file_path)

            test_df = DataValidation.read_data(self, self.data_ingestion_artifact.test_file_path)

            # Validate the data against the schema
            if not self.validate_schema(train_df):
                data_validation_logger.error("Data validation failed for training data")
                raise ChurnPredictionException("Data validation failed for training data", sys)

            if not self.validate_schema(test_df):
                data_validation_logger.error("Data validation failed for test data")
                raise ChurnPredictionException("Data validation failed for test data", sys)
            
            if not self.check_data_types(train_df):
                data_validation_logger.error("Data type validation failed for training data")
                raise ChurnPredictionException("Data type validation failed for training data", sys)
            
            if not self.check_data_types(test_df):
                data_validation_logger.error("Data type validation failed for test data")
                raise ChurnPredictionException("Data type validation failed for test data", sys)
            
            status = self.check_data_drift(train_df, test_df)
            os.makedirs(self.data_validation_config.invalid_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.validated_dir, exist_ok=True)
            
            if not status:
                data_validation_logger.error("Data drift detected between training and test data")
                
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                raise ChurnPredictionException("Data drift detected between training and test data", sys)
            
            else:
                
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)
                data_validation_logger.info("Data validation successful, no data drift detected")
            

            
            

            # If validation is successful, create a DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            data_validation_logger.info("Data validation completed successfully")
            return data_validation_artifact

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    