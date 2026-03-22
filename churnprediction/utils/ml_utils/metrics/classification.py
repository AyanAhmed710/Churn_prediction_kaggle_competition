import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score ,recall_score ,precision_score ,roc_auc_score
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.entity.artifact import ClassificationArtifact


def get_classification(y_test ,y_pred):
    try:
        f1 = f1_score(y_test ,y_pred)
        recall = recall_score(y_test ,y_pred)
        precision = precision_score(y_test ,y_pred)
        roc = roc_auc_score(y_test ,y_pred)
        return ClassificationArtifact(f1_score=f1 ,recall=recall ,precision=precision ,roc_auc=roc)
    except Exception as e:
        raise ChurnPredictionException(e, sys)
    

def evaluate_model(X_train, Y_train, X_test, Y_test, models, params):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            
            param_grid = params[model_name]

            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, Y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_f1 = f1_score(Y_train, y_train_pred)
            test_f1 = f1_score(Y_test, y_test_pred)

            report[model_name] = test_f1
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise ChurnPredictionException(e, sys)