import logging
from pathlib import Path
import numpy as np


def set_up_logger(logger, verbosity_level, loglevel, logfile: Path) -> None:
    """
    Set up both a stream handler that outputs stuff on
    stdout according to the verbosity as well as a file handler that writes to
    a file according to the loglevel set
    """
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(logfile.absolute())
    c_handler.setLevel(logging.ERROR - (verbosity_level * 10))
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


def select_ftdc_dist_from_ctdc_bin(full_hist: list, ctdc_code: int, density: bool = True):
    """
    create an histogram of ftdc codes that from the full toa histogram
    """
    assert ctdc_code < 32
    assert ctdc_code >= 0
    assert len(full_hist) == 1024
    ctdc_hist = sum([np.array(full_hist[i*256:(i+1)*256])
                    for i in range(1024//256)])
    ftdc_hist = np.array(ctdc_hist[ctdc_code*8:(ctdc_code+1)*8])
    return ftdc_hist / sum(ftdc_hist)


def ensure_channel_output_dir(output_dir: Path, tdc_type: str, tdc_summary_data: dict) -> Path:
    """
    Make sure that the directory that we want to write the plots into exists
    """
    chip = tdc_summary_data['chip']
    half = tdc_summary_data['half']
    chan = tdc_summary_data['ch']
    chip_dir = output_dir / f'chip_{chip}'
    half_dir = chip_dir / f'half_{half}'
    chan_dir = half_dir / f'chan_{chan}'
    if not chip_dir.exists():
        chip_dir.mkdir()
    if not half_dir.exists():
        half_dir.mkdir()
    if not chan_dir.exists():
        chan_dir.mkdir()
    if tdc_type == 'ftdc':
        tdc_dir = chan_dir / 'FTDC'
        if not tdc_dir.exists():
            tdc_dir.mkdir()
    elif tdc_type == 'ctdc':
        tdc_dir = chan_dir / 'CTDC'
        if not tdc_dir.exists():
            tdc_dir.mkdir()
    else:
        tdc_dir = chan_dir
    return tdc_dir


def prepare_data_for_ctdc_comparison(tdc_summary_data: dict, FTDC_codes: int = 2**3) -> list:
    full_histogram = tdc_summary_data['full_hist']
    bin_count = len(full_histogram)

    # check inputs to be of the power of 2
    assert bin_count in [2**i for i in range(1, 11)]
    assert FTDC_codes in [2**i for i in range(10)]

    course_bin_count = bin_count // FTDC_codes
    # here we sum over the bottom 3 bits so merging groups of 8 bins into 1
    course_histogram = np.array(
        [sum(full_histogram[i*FTDC_codes:(i+1)*FTDC_codes]) for i in range(course_bin_count)])

    section_len = course_bin_count // 4
    # now we cut the histogram into the four sections
    sections = [
        course_histogram[i*section_len:(i+1)*section_len] for i in range(4)]
    for i, section in enumerate(sections):
        sections[i] = np.array([x if x != 0 else 1 for x in section])
    return sections
