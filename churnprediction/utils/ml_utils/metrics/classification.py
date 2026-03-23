import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score ,recall_score ,precision_score ,roc_auc_score
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.entity.artifact import ClassificationArtifact
import mlflow
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone



def get_classification(y_test ,y_pred):
    try:
        f1 = f1_score(y_test ,y_pred)
        recall = recall_score(y_test ,y_pred)
        precision = precision_score(y_test ,y_pred)
        roc = roc_auc_score(y_test ,y_pred)
        return ClassificationArtifact(f1_score=f1 ,recall=recall ,precision=precision ,roc_auc=roc)
    except Exception as e:
        raise ChurnPredictionException(e, sys)
    

# def evaluate_model(X_train, Y_train, X_test, Y_test, models, params):
#     try:
#         report = {}
#         trained_models = {}
        

#         for model_name, model in models.items():
            
#             param_grid = params[model_name]

#             gs = GridSearchCV(model, param_grid, cv=3)
#             gs.fit(X_train, Y_train)

#             best_model = gs.best_estimator_

#             y_train_pred = best_model.predict(X_train)
#             y_test_pred = best_model.predict(X_test)

#             train_f1 = f1_score(Y_train, y_train_pred)
#             test_f1 = f1_score(Y_test, y_test_pred)

#             report[model_name] = test_f1
#             trained_models[model_name] = best_model

#         return report, trained_models

#     except Exception as e:
#         raise ChurnPredictionException(e, sys)
    

def evaluate_model(X_train, Y_train, X_test, Y_test, models, params):
    try:
        report = {}
        trained_models = {}

        mlflow.set_experiment("ChurnPrediction_ModelSelection")

        with mlflow.start_run(run_name="model_selection_overview") as parent_run:

            mlflow.set_tag("stage",  "model_selection")
            mlflow.set_tag("author", "AYAN")

            for model_name, model in models.items():

                param_grid = params[model_name]
                gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1)

                with mlflow.start_run(run_name=model_name, nested=True):

                    gs.fit(X_train, Y_train)

                    # ✅ One nested run per param combo — params + accuracy logged
                    for i in range(len(gs.cv_results_['params'])):
                        with mlflow.start_run(run_name=f"{model_name}_combo_{i}", nested=True):
                            mlflow.log_params(gs.cv_results_['params'][i])
                            mlflow.log_metric("cv_mean_f1", gs.cv_results_["mean_test_score"][i])
                            mlflow.log_metric("cv_std_f1",  gs.cv_results_["std_test_score"][i])

                    # ✅ Best params + final metrics on model-level run
                    best_model     = gs.best_estimator_
                    y_train_pred   = best_model.predict(X_train)
                    y_test_pred    = best_model.predict(X_test)

                    mlflow.log_params(gs.best_params_)
                    mlflow.log_metric("best_cv_f1",     gs.best_score_)
                    mlflow.log_metric("train_f1",       f1_score(Y_train, y_train_pred))
                    mlflow.log_metric("test_f1",        f1_score(Y_test,  y_test_pred))
                    mlflow.log_metric("test_precision",  precision_score(Y_test, y_test_pred))
                    mlflow.log_metric("test_recall",     recall_score(Y_test,    y_test_pred))
                    mlflow.log_metric("test_roc_auc",    roc_auc_score(Y_test,   y_test_pred))
                    mlflow.set_tag("best_combo",         str(gs.best_params_))
                    mlflow.sklearn.log_model(best_model,  artifact_path=model_name)

                    report[model_name]         = f1_score(Y_test, y_test_pred)
                    trained_models[model_name] = best_model

            # ── Parent summary ─────────────────────────────────────────
            best_model_name  = max(report, key=report.get)
            best_model_score = report[best_model_name]

            for model_name, score in report.items():
                mlflow.log_metric(f"{model_name}_test_f1", score)

            mlflow.set_tag("best_model",       best_model_name)
            mlflow.set_tag("best_model_score", f"{best_model_score:.4f}")
            mlflow.log_param("winner",          best_model_name)

        return report, trained_models

    except Exception as e:
        raise ChurnPredictionException(e, sys)



