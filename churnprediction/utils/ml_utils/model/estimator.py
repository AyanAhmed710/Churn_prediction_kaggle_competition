import os
import sys
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import get_logger


class ChurnModel:
    def __init__(self , preprocessor ,model):
        self.preprocessor = preprocessor
        self.model = model

    def predict_old(self , data):
        try:
            return self.model.predict(data)
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
        

    def predict_new(self , data):
        try:
            return self.model.predict(self.preprocessor.transform(data))
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)