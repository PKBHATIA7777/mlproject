import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


# ═══════════════════════════════════════════════════════════════
# CLASS 1: PredictPipeline
# ───────────────────────────────────────────────────────────────
# This class is responsible for doing the actual prediction.
# Think of it as the "engine room" — it loads the saved model
# and preprocessor and uses them to predict the math score.
# ═══════════════════════════════════════════════════════════════
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Takes the user's input (as a DataFrame) and returns the predicted score.

        Steps:
        1. Load the saved model (model.pkl) from the artifacts folder
        2. Load the saved preprocessor (preprocessor.pkl) — this is the
           pipeline that does One Hot Encoding + Standard Scaling
        3. Transform/scale the input data using the preprocessor
           (same way the training data was scaled — model needs same format!)
        4. Run the model on the scaled data and return the prediction
        """
        try:
            # Path to our saved model and preprocessor
            # These .pkl files were created during the Model Trainer step
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            # load_object() simply unpickles (loads) the saved file
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Apply the same preprocessing the training data went through:
            # → One Hot Encode categorical columns (gender, lunch, etc.)
            # → Standard Scale numeric columns (reading_score, writing_score)
            # This is critical! Model was trained on scaled data, so
            # our input must also be scaled the same way.
            data_scaled = preprocessor.transform(features)

            # Now feed the scaled data into the model to get the prediction
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


# ═══════════════════════════════════════════════════════════════
# CLASS 2: CustomData
# ───────────────────────────────────────────────────────────────
# This class acts like a "form data container".
# When the user fills the HTML form and hits Predict,
# app.py collects each field and stores it here in this object.
#
# Think of it as a box with labelled compartments:
#   [ gender | race | education | lunch | test_course | reading | writing ]
#
# Later we convert this box into a pandas DataFrame (like a table)
# because that is what the model expects as input.
# ═══════════════════════════════════════════════════════════════
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        # Store each value from the form as an attribute of this object
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the stored form values into a pandas DataFrame.

        Why? Because our ML model was trained on a DataFrame.
        So the input at prediction time must also be a DataFrame
        with the exact same column names.

        Result looks like this (1 row table):
        | gender | race_ethnicity | parental_level_of_education | lunch | test_preparation_course | reading_score | writing_score |
        | male   | group B        | bachelor's degree           | standard | none                 | 72            | 74            |
        """
        try:
            # Create a dictionary where keys = column names, values = user's input
            # Each value is wrapped in a list [] because DataFrame expects lists
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary into a proper pandas DataFrame and return it
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)