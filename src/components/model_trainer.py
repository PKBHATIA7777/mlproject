
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


# This class stores the path where the trained model will be saved
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        # Create the configuration object
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Separate input features (X) and target values (y)
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Create different regression models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            # Train and evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            # Find the highest R2 score
            best_model_score = max(model_report.values())

            # Find the name of the model with the highest score
            best_model_name = max(model_report, key=model_report.get)

            # Get the best model using its name
            best_model = models[best_model_name]

            # Check whether the best model score is good enough
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model: {best_model_name}, "
                f"R2 Score: {best_model_score}"
            )

            # Save the best trained model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Predict the target values using the best model
            predicted = best_model.predict(X_test)

            # Calculate the final R2 score on test data
            r2_square = r2_score(y_test, predicted)

            # Return the final model score
            return r2_square

        except Exception as e:
            # Log the error and raise it again
            logging.error(f"Error occurred during model training: {e}")
            raise CustomException(e)
