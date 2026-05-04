
import logging
from pathlib import Path
from utils import get_project_folder

""""
    Niveles de gravedad que tiene los logRecord

        CRITICAL
        ERROR
        WARNING
        INFO
        DEBUG
"""

def setup_logging(nivel: str, log_filename: str ="app.log"):

    path_proyecto = get_project_folder()
    Path(path_proyecto/"logs").mkdir(exist_ok=True)

    fichero_log = path_proyecto / "logs" / log_filename

    logging.basicConfig(
        level = getattr(logging, nivel.upper(), logging.DEBUG),
        format = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)s | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(), # stdout
            logging.FileHandler(fichero_log)
        ]
    )
