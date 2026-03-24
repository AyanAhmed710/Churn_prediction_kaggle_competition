import os
import sys



#Define constants for training pipeline

PIPELINE_NAME="churn_prediction_pipeline"
ARTIFACT_NAME = "Artifact"
TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"
TARGET_COLUMN="Churn"
FILE_NAME = "churn.csv"

SCHEMA_FILE_PATH=os.path.join("data_schema", "schema.yaml")


#Define constants for data ingestion
DATA_INGESTION_DIR_NAME="data_ingestion"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION=0.2
DATA_INGESTION_FEATURE_STORE_DIR="feature_store"
DATA_INGESTION_INGESTED_DIR="ingested"
DATA_INGESTION_DATABASE_NAME="churn_prediction"
DATA_INGESTION_COLLECTION_NAME="customer_churn_data"
DATA_INGESTION_RANDOM_STATE=42

#Define constants for data validation
DATA_VALIDATION_DIR_NAME="data_validation"
DATA_VALIDATION_VALIDATED_DIR="validated"
DATA_VALIDATION_INVALID_DIR="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR="drift_report"
DATA_VALIDATION_DRIFT_FILE_NAME="report.yaml"

#Define constants for data transformation 
DATA_TRANSFORMATION_DIR_NAME="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR="transformed"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME="transformed_train.npy"
DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME="transformed_test.npy"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_NAME="transformed_object.pkl"

SMOTE_PARAMERTERS :dict = {
    "k_neighbors" : 5,
    "random_state" : 42,
    "sampling_strategy" : "auto"
}

INTERNET_SERVICE_COLS = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

BINARY_YES_NO_COLS = [
    'Partner', 'Dependents', 'PhoneService',
    'PaperlessBilling', 'Churn'
]


#Define Constants for Model Training

MODEL_TRAINER_DIR_NAME="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR="trained_model"
MODEL_TRAINING_REPORT ="report.yaml"
MODEL_TRAINER_TRAINED_MODEL_NAME="model.pkl"

FINAL_PUSHED_MODEL_PATH=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\Final_Model\model.pkl"

FINAL_PUSHED_PREPROCESSOR_PATH=r"D:\MLOPS\Churn_Prediction\Churn_prediction_kaggle_competition\Final_Model\preprocessor.pkl"

BUCKET_NAME="churnprediction123"



