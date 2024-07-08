import logging
import os

def get_logger(name: str) -> logging.Logger:
    # Ensure the logs directory exists
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, 'app.log')),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger(name)
    return logger