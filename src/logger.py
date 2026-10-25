import logging
import os
from datetime import datetime

# Create a unique log file name using the current date and time.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create a folder named 'logs' in the current working directory.
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Full path to the current log file.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logger so messages are saved in the log file.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)

# Logger object for the application to use.
logger = logging.getLogger(__name__)

# Run this block only when the file is executed directly.
if __name__ == "__main__":
    # Log a message to confirm the logger is active.
    logging.info("Logging has started")

