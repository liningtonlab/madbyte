import sys
import logging
from pathlib import Path

LOGS = {}

def get_logger(name):
    """Helper function to get logger"""
    try:
        logger = LOGS[name]
    except:
        logger = logging.getLogger(name)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(level)
        LOGS[name] = logger
    finally:
        return logger
    

def setup_logging(fname, fpath=None, level=logging.DEBUG):
    """Setup logging to file and optionally STDOUT

    Args:
        fname (str): log filename
        fpath (str): log path default to currect dir
        verbose (bool): Log to STDOUT (True) or not (False)
        level (logging.LEVEL): Logger level
    """

    logger = logging.getLogger("MADByTE")
    fpath = Path(fpath) if fpath else Path()
    fname = fpath.joinpath(fname)

    fh = logging.FileHandler(fname, mode='w')
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s: %(levelname)s : %(message)s"
    )
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)
