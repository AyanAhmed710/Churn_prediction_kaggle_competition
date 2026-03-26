# import os
# import pandas as pd
# import numpy as np
# from churnprediction.exception.exception import ChurnPredictionException
# from churnprediction.logging.logger import get_logger
# from churnprediction.entity.config_entity import DataTransformationConfig
# from churnprediction.entity.artifact import DataValidationArtifact , DataTransformationArtifact
# from churnprediction.utils import save_numpy ,save_object
# from churnprediction.constants.training_pipeline import TARGET_COLUMN ,SMOTE_PARAMERTERS ,INTERNET_SERVICE_COLS ,BINARY_YES_NO_COLS
# from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from churnprediction.utils.ml_utils.model.preprocessor_utils import ManualEncoder
# import sys


# data_transformation_logger = get_logger("data_transformation")

# class DataTransformation:
#     def __init__(self, data_transformation_config: DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
#         try:
#             self.data_transformation_config :DataTransformationConfig = data_transformation_config
#             self.data_validation_artifact :DataValidationArtifact = data_validation_artifact
#         except Exception as e:
#             raise ChurnPredictionException(e, sys)
        

#     def read_data(self ,file_path):
#         try:
#             data_transformation_logger.info(f"Reading data from {file_path}")
#             data = pd.read_csv(file_path)
#             print(data.columns.tolist())   # check exact column names
#             print(data.dtypes) 
#             data_transformation_logger.info(f"Data read successfully from {file_path}")
#             return data
#         except Exception as e:
#             raise ChurnPredictionException(e, sys)
        
    
        
#     def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame: 
#          # fix 'sefl' typo too
#      try:
#         data_transformation_logger.info("Starting preprocessing")

#         # Step 1 — collapse No internet service → No
#         for col in INTERNET_SERVICE_COLS:
#             df[col] = df[col].replace({'No internet service': 'No'})

#         # Step 2 — binary encode internet-dependent cols
#         for col in INTERNET_SERVICE_COLS:
#             df[col] = df[col].map({'Yes': 1, 'No': 0})

#         # Step 3 — binary encode other Yes/No cols
#         for col in BINARY_YES_NO_COLS:
#             df[col] = df[col].map({'Yes': 1, 'No': 0})

#         # Step 4 — MultipleLines separately (has 3 values)
#         df['MultipleLines'] = df['MultipleLines'].map(
#             {'Yes': 1, 'No': 0, 'No phone service': 0}
#         )

#         data_transformation_logger.info("Preprocessing completed successfully")
#         return df  # always return the df

#      except Exception as e:
#         raise ChurnPredictionException(e, sys)
        
        
#     def get_transformer(self):
#         try:
#             data_transformation_logger.info("Getting transformer")

#             ohe =OneHotEncoder()
#             std=StandardScaler()
#             preprocessor = ColumnTransformer(
#                 [
#                     ("ohe", ohe, ["gender", "InternetService", "PaymentMethod", "Contract"]),
#                     ("std", std, ["tenure", "MonthlyCharges", "TotalCharges"]),
#                 ] ,remainder="passthrough"
#             )

#             full_pipeline = Pipeline([
#             ("manual_encoder", ManualEncoder()),
#             ("preprocessor", preprocessor),
#         ])

#             return full_pipeline
            
#         except Exception as e:
#             raise ChurnPredictionException(e, sys)
        
#     def resampling(self ,X_train ,y_train):
#         try:
#             data_transformation_logger.info("Starting resampling")
#             smote=SMOTE(**SMOTE_PARAMERTERS)
#             X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#             data_transformation_logger.info("Resampling completed successfully")
#             return X_train_res , y_train_res
#         except Exception as e:
#             raise ChurnPredictionException(e, sys)
        
#     def initiate_data_transformation(self):
#         try:
#             data_transformation_logger.info("Starting data transformation process")
#             train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
#             test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
#             train_df.drop(columns=[ "id"], inplace=True)
#             test_df.drop(columns=["id"], inplace=True)

#             # preprocess_train_df=self.preprocessing(train_df)
#             # preprocess_test_df=self.preprocessing(test_df)

#             transformer =self.get_transformer()

#             X_train_features_df = train_df.drop(TARGET_COLUMN, axis=1)
#             X_test_features_df = test_df.drop(TARGET_COLUMN, axis=1)
#             Y_train_df = train_df[TARGET_COLUMN].map({'Yes': 1, 'No': 0}).astype('int64')
#             Y_test_df = test_df[TARGET_COLUMN].map({'Yes': 1, 'No': 0}).astype('int64')

#             preprocess_X_train_features_df = transformer.fit_transform(X_train_features_df)
#             preprocess_X_test_features_df = transformer.transform(X_test_features_df)


#             X_train_res ,Y_train_res = self.resampling(preprocess_X_train_features_df, Y_train_df)
            
#             train_arr =np.c_[X_train_res, Y_train_res]
#             test_arr=np.c_[preprocess_X_test_features_df, Y_test_df]

            

#             save_numpy(
#                 file_path=self.data_transformation_config.transformed_train_file_path,
#                 array=train_arr
#             )

#             save_numpy(
#                 file_path=self.data_transformation_config.transformed_test_file_path,
#                 array=test_arr
#             )

#             save_object(
#                 file_path=self.data_transformation_config.transformed_object_file_path,
#                 object=transformer
#             )

#             data_transformation_artifact = DataTransformationArtifact(
#                 transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                 transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
#                 transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
#             )

#             # --- Save as CSV for inspection ---
#             # Get column names after ColumnTransformer
#             ohe_cols = transformer.named_transformers_['ohe'].get_feature_names_out(
#                 ["gender", "InternetService", "PaymentMethod", "Contract"]
#             ).tolist()
#             std_cols  = ["tenure", "MonthlyCharges", "TotalCharges"]
#             remainder_cols = [col for col in X_train_features_df.columns 
#                             if col not in ["gender", "InternetService", "PaymentMethod", 
#                                             "Contract", "tenure", "MonthlyCharges", "TotalCharges"]]

#             all_feature_cols = ohe_cols + std_cols + remainder_cols

#             train_df_to_save = pd.DataFrame(train_arr, columns=all_feature_cols + [TARGET_COLUMN])
#             test_df_to_save  = pd.DataFrame(test_arr,  columns=all_feature_cols + [TARGET_COLUMN])

#             train_df_to_save.to_csv(
#                 self.data_transformation_config.transformed_train_file_path.replace('.npy', '.csv'),
#                 index=False
#             )
#             test_df_to_save.to_csv(
#                 self.data_transformation_config.transformed_test_file_path.replace('.npy', '.csv'),
#                 index=False
#             )

#             data_transformation_logger.info("Saved transformed data as CSV for inspection")

#             data_transformation_logger.info("Data transformation completed successfully")

#             return data_transformation_artifact

                


#         except Exception as e:
#             raise ChurnPredictionException(e, sys)




import os
import pandas as pd
import numpy as np
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import get_logger
from churnprediction.entity.config_entity import DataTransformationConfig
from churnprediction.entity.artifact import DataValidationArtifact, DataTransformationArtifact
from churnprediction.utils import save_numpy, save_object
from churnprediction.constants.training_pipeline import TARGET_COLUMN, SMOTE_PARAMERTERS, INTERNET_SERVICE_COLS, BINARY_YES_NO_COLS
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from churnprediction.utils.ml_utils.model.preprocessor_utils import ManualEncoder
import sys
 
 
data_transformation_logger = get_logger("data_transformation")
 
 
class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
        except Exception as e:
            raise ChurnPredictionException(e, sys)
 
    def read_data(self, file_path):
        try:
            data_transformation_logger.info(f"Reading data from {file_path}")
            data = pd.read_csv(file_path)
            data_transformation_logger.info(f"Data read successfully from {file_path}")
            return data
        except Exception as e:
            raise ChurnPredictionException(e, sys)
 
    def get_transformer(self):
        try:
            data_transformation_logger.info("Getting transformer")
 
            ohe = OneHotEncoder()
            std = StandardScaler()
 
            preprocessor = ColumnTransformer(
                [
                    ("ohe", ohe, ["gender", "InternetService", "PaymentMethod", "Contract"]),
                    ("std", std, ["tenure", "MonthlyCharges", "TotalCharges"]),
                ],
                remainder="passthrough"
            )
 
            full_pipeline = Pipeline([
                ("manual_encoder", ManualEncoder()),
                ("preprocessor", preprocessor),
            ])
 
            return full_pipeline
 
        except Exception as e:
            raise ChurnPredictionException(e, sys)
 
    def resampling(self, X_train, y_train):
        try:
            data_transformation_logger.info("Starting resampling")
            smote = SMOTE(**SMOTE_PARAMERTERS)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            data_transformation_logger.info("Resampling completed successfully")
            return X_train_res, y_train_res
        except Exception as e:
            raise ChurnPredictionException(e, sys)
 
    def initiate_data_transformation(self):
        try:
            data_transformation_logger.info("Starting data transformation process")
 
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
 
            train_df.drop(columns=["id"], inplace=True)
            test_df.drop(columns=["id"], inplace=True)
 
            # Get transformer (ManualEncoder + ColumnTransformer inside Pipeline)
            transformer = self.get_transformer()
 
            # Split features and target
            X_train_features_df = train_df.drop(TARGET_COLUMN, axis=1)
            X_test_features_df = test_df.drop(TARGET_COLUMN, axis=1)
 
            # Map target Yes/No → 1/0
            Y_train_df = train_df[TARGET_COLUMN].map({'Yes': 1, 'No': 0}).astype('int64')
            Y_test_df = test_df[TARGET_COLUMN].map({'Yes': 1, 'No': 0}).astype('int64')
 
            # Fit on train, transform both
            preprocess_X_train_features_df = transformer.fit_transform(X_train_features_df)
            preprocess_X_test_features_df = transformer.transform(X_test_features_df)
 
            # SMOTE resampling
            X_train_res, Y_train_res = self.resampling(preprocess_X_train_features_df, Y_train_df)
 
            # Combine features + target into arrays
            train_arr = np.c_[X_train_res, Y_train_res]
            test_arr = np.c_[preprocess_X_test_features_df, Y_test_df]
 
            # Save numpy arrays
            save_numpy(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
 
            # Save transformer pipeline
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                object=transformer
            )
 
            # ── Save as CSV for inspection ──────────────────────────────────
            # Access ColumnTransformer inside the Pipeline
            column_transformer = transformer.named_steps['preprocessor']
 
            ohe_cols = column_transformer.named_transformers_['ohe'].get_feature_names_out(
                ["gender", "InternetService", "PaymentMethod", "Contract"]
            ).tolist()
 
            std_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
 
            remainder_cols = [
                col for col in X_train_features_df.columns
                if col not in ["gender", "InternetService", "PaymentMethod",
                               "Contract", "tenure", "MonthlyCharges", "TotalCharges"]
            ]
 
            all_feature_cols = ohe_cols + std_cols + remainder_cols
 
            train_df_to_save = pd.DataFrame(train_arr, columns=all_feature_cols + [TARGET_COLUMN])
            test_df_to_save = pd.DataFrame(test_arr, columns=all_feature_cols + [TARGET_COLUMN])
 
            train_df_to_save.to_csv(
                self.data_transformation_config.transformed_train_file_path.replace('.npy', '.csv'),
                index=False
            )
            test_df_to_save.to_csv(
                self.data_transformation_config.transformed_test_file_path.replace('.npy', '.csv'),
                index=False
            )
 
            data_transformation_logger.info("Saved transformed data as CSV for inspection")
            data_transformation_logger.info("Data transformation completed successfully")
 
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
 
        except Exception as e:
            raise ChurnPredictionException(e, sys)
