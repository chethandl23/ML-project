import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.logger import logging
from src.exceptions import CustomException
import dill

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        model_report = {}
        for model_name , model in models.items():
            model.fit(x_train, y_train)
            y_test_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)
            model_test_r2_score = r2_score(y_test, y_test_pred)
            model_train_r2_score = r2_score(y_train, y_train_pred)
            model_report[model_name] = {
                "train_r2_score": model_train_r2_score,
                "test_r2_score": model_test_r2_score
            }
        return model_report
    except Exception as e:
        logging.error(f"Error occurred while evaluating models: {e}")
        raise CustomException(e, sys)