import os
import sys

from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import get_logger
from churnprediction.entity.config_entity import ModelTrainingConfig
from churnprediction.entity.artifact import ModelTrainerArtifact
from churnprediction.entity.artifact import DataTransformationArtifact
from churnprediction.utils.ml_utils.metrics.classification import get_classification ,evaluate_model
from churnprediction.utils.ml_utils.model.estimator import ChurnModel
from churnprediction.utils import load_object ,write_yaml_file ,read_numpy_array ,save_object
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score ,recall_score ,precision_score ,roc_auc_score
import mlflow 
import dagshub


class ModelTrainer:
    def __init__(self , model_trainer_config : ModelTrainingConfig , data_transformation_artifact : DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ChurnPredictionException(e, sys)
        
    # def train_model(self ,X_train ,X_test ,Y_train ,Y_test):

    #     os.makedirs(os.path.dirname(self.model_trainer_config.report_file_path), exist_ok=True)

    #     models = {
    #             "LogisticRegression": LogisticRegression(),
    #             "RandomForestClassifier": RandomForestClassifier()
    #             # You can skip KNeighbors for now if you want it even faster
    #         }

    #     params = {
    #             "LogisticRegression": {
    #                 "solver": ["liblinear"],  # safe solver
    #                 "penalty": ["l2"],        # compatible with liblinear
    #                 "C": [1]                  # just one value
    #             },
    #             "RandomForestClassifier": {
    #                 "n_estimators": [10],     # very small forest
    #                 "criterion": ["gini"]     # one criterion
    #             }
    #         }

    #     report, trained_models = evaluate_model(X_train ,Y_train ,X_test ,Y_test ,models ,params)

    #     best_model_name = max(report, key=report.get)
    #     best_model = trained_models[best_model_name]



    #     write_yaml_file(self.model_trainer_config.report_file_path , report)

        
        

    #     train_artifact =get_classification(Y_train ,best_model.predict(X_train))
    #     test_artifact=get_classification(Y_test ,best_model.predict(X_test))

    #     preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

    #     churn_model = ChurnModel(preprocessor ,best_model)

    #     os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

    #     save_object(file_path=self.model_trainer_config.trained_model_file_path , object=churn_model)

    #     return ModelTrainerArtifact(model_file_path=self.model_trainer_config.trained_model_file_path , train_artifact=train_artifact, test_artifact=test_artifact)

    # def train_model(self, X_train, X_test, Y_train, Y_test):

    #     os.makedirs(os.path.dirname(self.model_trainer_config.report_file_path), exist_ok=True)

    #     models = {
    #         "LogisticRegression":    LogisticRegression(),
    #         "RandomForestClassifier": RandomForestClassifier(),
    #     }

    #     params = {
    #         "LogisticRegression": {
    #             "solver":  ["liblinear"],
    #             "penalty": ["l1", "l2"],       # ← 2 options
    #             "C":       [0.1, 1, 10],       # ← 3 options = 6 trials total
    #         },
    #         "RandomForestClassifier": {
    #             "n_estimators": [10, 50, 100], # ← 3 options
    #             "criterion":    ["gini", "entropy"], # ← 2 options = 6 trials total
    #         },
    #     }

    #     report, trained_models = evaluate_model(X_train, Y_train, X_test, Y_test, models, params)

    #     best_model_name  = max(report, key=report.get)
    #     best_model       = trained_models[best_model_name]

    #     write_yaml_file(self.model_trainer_config.report_file_path, report)

    #     train_artifact = get_classification(Y_train, best_model.predict(X_train))
    #     test_artifact  = get_classification(Y_test,  best_model.predict(X_test))

    #     # ── Final run: log the chosen production model ───────────────────────
    #     mlflow.set_experiment("ChurnPrediction_ModelSelection")

    #     with mlflow.start_run(run_name=f"BEST_MODEL__{best_model_name}"):

    #         mlflow.set_tag(" best_model",  best_model_name)
    #         mlflow.set_tag("stage",          "production_candidate")

    #         mlflow.log_metric("train_f1",       train_artifact.f1_score)
    #         mlflow.log_metric("train_precision", train_artifact.precision)
    #         mlflow.log_metric("train_recall",    train_artifact.recall)

    #         mlflow.log_metric("test_f1",        test_artifact.f1_score)
    #         mlflow.log_metric("test_precision",  test_artifact.precision)
    #         mlflow.log_metric("test_recall",     test_artifact.recall)

    #         mlflow.log_artifact(self.model_trainer_config.report_file_path, artifact_path="reports")

    #         mlflow.sklearn.log_model(
    #             sk_model=best_model,
    #             artifact_path="production_model",
    #             registered_model_name=f"ChurnPrediction_{best_model_name}",
    #         )

    #     preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
    #     churn_model  = ChurnModel(preprocessor, best_model)

    #     os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
    #     save_object(file_path=self.model_trainer_config.trained_model_file_path, object=churn_model)

    #     return ModelTrainerArtifact(
    #         model_file_path=self.model_trainer_config.trained_model_file_path,
    #         train_artifact=train_artifact,
    #         test_artifact=test_artifact,
    #     )   

    def train_model(self, X_train, X_test, Y_train, Y_test):

        os.makedirs(os.path.dirname(self.model_trainer_config.report_file_path), exist_ok=True)

        models = {
            "LogisticRegression":     LogisticRegression(),
            "RandomForestClassifier": RandomForestClassifier(),
        }

        params = {
            "LogisticRegression": {
                "solver":  ["liblinear"],
                "penalty": ["l1", "l2"],
                "C":       [0.1, 1, 10],
            },
            "RandomForestClassifier": {
                "n_estimators": [10, 50, 100],
                "criterion":    ["gini", "entropy"],
            },
        }

        report, trained_models = evaluate_model(X_train, Y_train, X_test, Y_test, models, params)

        best_model_name  = max(report, key=report.get)
        best_model       = trained_models[best_model_name]

        write_yaml_file(self.model_trainer_config.report_file_path, report)

        train_artifact = get_classification(Y_train, best_model.predict(X_train))
        test_artifact  = get_classification(Y_test,  best_model.predict(X_test))

        mlflow.set_experiment("ChurnPrediction_ModelSelection")

        with mlflow.start_run(run_name=f"BEST_MODEL__{best_model_name}"):

            # ✅ Log all params of the final best model
            for param_name, param_value in best_model.get_params().items():
                mlflow.log_param(param_name, param_value)

            # ✅ Log the grid that was searched for this model
            for param_name, param_values in params[best_model_name].items():
                mlflow.log_param(f"grid_{param_name}", str(param_values))

            mlflow.log_param("model_name", best_model_name)
            mlflow.set_tag("stage",        "production_candidate")
            mlflow.set_tag("best_model",    best_model_name)

            mlflow.log_metric("train_f1",        train_artifact.f1_score)
            mlflow.log_metric("train_precision",  train_artifact.precision)
            mlflow.log_metric("train_recall",     train_artifact.recall)
            mlflow.log_metric("test_f1",          test_artifact.f1_score)
            mlflow.log_metric("test_precision",   test_artifact.precision)
            mlflow.log_metric("test_recall",      test_artifact.recall)

            mlflow.log_artifact(self.model_trainer_config.report_file_path, artifact_path="reports")

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="production_model",
                registered_model_name="ChurnPrediction",
            )

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        churn_model  = ChurnModel(preprocessor, best_model)

        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, object=churn_model)

        return ModelTrainerArtifact(
            model_file_path=self.model_trainer_config.trained_model_file_path,
            train_artifact=train_artifact,
            test_artifact=test_artifact,
        )





    def initialize_model_training(self):
        try:
            train_arr = read_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = read_numpy_array(self.data_transformation_artifact.transformed_test_file_path)
            X_train , Y_train = train_arr[:,:-1] , train_arr[:,-1]
            X_test , Y_test = test_arr[:,:-1] , test_arr[:,-1]

            model_training_artifact = self.train_model(X_train ,X_test ,Y_train ,Y_test)


            return model_training_artifact


        except Exception as e:
            raise ChurnPredictionException(e, sys)
