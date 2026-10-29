
# Import libraries for working with files and data
import os
import sys
import numpy as np
import pandas as pd

# Import tools from Scikit-learn for preprocessing data
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import our custom exception and logging functions
from src.exception import CustomException
from src.logger import logging

# Import dataclass to store file paths
from dataclasses import dataclass

from src.utils import save_object


# ============================================================
# PART 1: Store the path where the preprocessor will be saved
# ============================================================

@dataclass
class DataTransformationConfig:
    # This is the location where the preprocessor object will be saved
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


# ============================================================
# PART 2: Create the DataTransformation class
# ============================================================

class DataTransformation:

    def __init__(self):
        # Load the configuration containing the file path
        self.data_transformation_config = DataTransformationConfig()


    # ========================================================
    # PART 3: Create pipelines to clean and transform data
    # ========================================================

    def get_data_transformer_object(self):
        try:
            # These are the numerical columns in our dataset
            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            # These are the categorical columns in our dataset
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Log the columns so we can check them later
            logging.info(
                f"Categorical columns: {categorical_columns}"
            )

            logging.info(
                f"Numerical columns: {numerical_columns}"
            )


            # =================================================
            # PART 4: Numerical pipeline
            # =================================================

            # Step 1: Fill missing numerical values with the median
            # Step 2: Scale numerical values to a similar range
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )


            # =================================================
            # PART 5: Categorical pipeline
            # =================================================

            # Step 1: Fill missing categorical values with the
            # most frequently occurring value
            # Step 2: Convert categories into numerical columns
            # Step 3: Scale the encoded values
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    ),
                    (
                        "scaler",
                        StandardScaler(with_mean=False)
                    )
                ]
            )


            # =================================================
            # PART 6: Combine both pipelines
            # =================================================

            # Apply the numerical pipeline to numerical columns
            # Apply the categorical pipeline to categorical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            logging.info("Numerical and categorical pipelines created")

            # Return the complete preprocessing object
            return preprocessor


        except Exception as e:
            # If an error occurs, raise our custom exception
            raise CustomException(e, sys)




    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Step 1: Read the training and testing CSV files
            
            # Step 1: Read the training and testing CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Step 2: Standardize column names
            # Convert names to lowercase, remove extra spaces,
            # and replace spaces and slashes with underscores
            train_df.columns = (
                train_df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
                .str.replace("/", "_", regex=False)
            )

            test_df.columns = (
                test_df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
                .str.replace("/", "_", regex=False)
            )

            # Step 3: Fix the inconsistent reading score column name
            train_df.rename(
                columns={"readingscore": "reading_score"},
                inplace=True
            )

            test_df.rename(
                columns={"readingscore": "reading_score"},
                inplace=True
            )

            # Step 4: Print the corrected column names for verification
            print("Training columns:", train_df.columns.tolist())
            print("Testing columns:", test_df.columns.tolist())

            logging.info("Training and testing data loaded successfully")


            logging.info("Read train and test data completed")

            # Step 2: Create the preprocessing object
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            # Step 3: Define the target column we want to predict
            target_column_name = "math_score"

            # Step 4: Separate input features (X) and target (y)
            # X contains all columns except math_score
            # y contains only math_score

            input_feature_train_df = train_df.drop(
                columns=[target_column_name],
            
            )
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name],
              
            )
            target_feature_test_df = test_df[target_column_name]

            # Step 5: Fit the preprocessor on training data and transform it
            # Transform test data using the same fitted preprocessor
            logging.info(
                "Applying preprocessing object on training and testing data"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Step 6: Combine the transformed input features with the target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # Step 7: Save the fitted preprocessing object for future use
            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed successfully")

            # Step 8: Return transformed train data, test data, and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            # Raise a custom exception if something goes wrong
            raise CustomException(e, sys)
