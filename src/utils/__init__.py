from .logger import Logger

logger = None

def init(log_dir, log_name):
    global logger
    logger = Logger(log_dir, log_name)
    
    