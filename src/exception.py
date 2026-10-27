import sys

from src.logger import logger

def error_message_detail(error, error_detail: sys):
    # Get the traceback details.
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name and line number where the error happened.
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    # Create a clear error message.
    error_message = (
        "Error occurred in python script name [{0}] line number [{1}] "
        "error message [{2}]"
    ).format(file_name, line_number, str(error))

    return error_message


class CustomException(Exception):
    """Exception with extra file and line details."""

    def __init__(self, error_message, error_detail: sys):
        # Keep the original message and add more detail.
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        # Show the full detailed message.
        return self.error_message