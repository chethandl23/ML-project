import logging
import os
from datetime import datetime

# Create logs directory
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create log file name with timestamp
LOG_FILE = f"log_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Full path for the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    logging.info("Logging setup complete.")
