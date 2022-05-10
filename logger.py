import logging
import os
import sys
from time import strftime
from constants import ROOT_DIR
def setup_logger(args):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    a_logger = logging.getLogger()
    a_logger.setLevel(args.log_level)
    log_dir=os.path.join(ROOT_DIR,"logs","output_logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    output_file_handler = logging.FileHandler(os.path.join(log_dir,strftime("log_%d_%m_%Y_%H_%M.log")))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    a_logger.propagate=False
    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)