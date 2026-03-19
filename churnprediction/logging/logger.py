import os
import logging
from datetime import datetime   
from logging.handlers import RotatingFileHandler


LOG_DIR="logs" 

def get_logger(name):

    os.makedirs(LOG_DIR, exist_ok=True)

    log_file_path = os.path.join(LOG_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler=RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    console_handler=logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


