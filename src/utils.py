
# Import libraries for handling files and errors
import os
import sys
import dill

# Import our custom exception class
from src.exception import CustomException


# This function saves a Python object to a file
def save_object(file_path, obj):
    try:
        # Get the folder path where the file will be saved
        dir_path = os.path.dirname(file_path)

        # Create the folder if it does not already exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode
        with open(file_path, "wb") as file_obj:
            # Save the Python object into the file
            dill.dump(obj, file_obj)

    except Exception as e:
        # If an error occurs, raise our custom exception
        raise CustomException(e, sys)
