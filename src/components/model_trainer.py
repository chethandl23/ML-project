import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exceptions import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (train_array[:,:-1], 
                                            train_array[:,-1],
                                            test_array[:,:-1],
                                            test_array[:,-1]
                                            )
            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=False),
                "KNeighborsRegressor": KNeighborsRegressor()
            }

            model_report:dict=evaluate_models(x_train = X_train, y_train = y_train, x_test = X_test, y_test = y_test, models = models)
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_r2_score'])
            best_model_score = model_report[best_model_name]['test_r2_score']


            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            logging.info(f"Predictions made on test data")
            test_r2_score = r2_score(y_test, predicted)
            return test_r2_score
        

        except Exception as e:
            logging.error(f"Error occurred while training model: {e}")
            raise CustomException(e, sys)
