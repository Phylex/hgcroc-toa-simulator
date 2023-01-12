"""
Module containing functions to handle configurations of the toa
"""
from pathlib import Path
import logging


def generate_default_toa_config() -> dict:
    """
    generate a default config for the ToA simulator ready to either load it
    into the ToA class
    """
    toa_config = {}
    toa_config['clock_frequency'] = 160_000_000
    return toa_config


def set_up_logger(logger, verbosity_level, loglevel, logfile: Path) -> None:
    """
    Set up both a stream handler that outputs stuff on
    stdout according to the verbosity as well as a file handler that writes to
    a file according to the loglevel set
    """
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(logfile.absolute())
    c_handler.setLevel(verbosity_level)
    f_handler.setLevel(loglevel)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(message)s')
    f_format = logging.Formatter(
        '[%(asctime)s] %(levelname)-10s:%(name)-30s %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
