"""
Module containing the main command line applications that
are available to the user
"""
import logging
from pathlib import Path
from . import __version__
import click
import yaml
from .utils import generate_default_toa_config


log_level_dict = {'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}


@click.group()
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='make the program be more verbose about what it is doing')
@click.option('-l', '--logfile',
              type=click.Path(dir_okay=False, file_okay=True),
              default='./toa-simulation.log',
              help='specify the location of the logfile',
              show_default=True)
@click.option('-fl', '--loglevel',
              type=click.Choice(log_level_dict.keys(),
                                case_sensitive=False),
              default=logging.INFO,
              help='Set the loglevel for the log file')
@click.option('-c', '--config-file', type=click.File('r'),
              default=None,
              help='config file holding the configuration of the ToA')
@click.version_option(__version__)
@click.pass_context
def cli(ctx, verbose, logfile: click.Path, loglevel: int, config_file: click.File):
    """
    Simulate and explore the ToA as implemented in the HGCROCv3
    """
    # in this function we can do all the things that would need doing
    # for every invocation of the tool, but nothing other than delegating
    # the actual tasks to commands
    # logging.basicConfig(filename="log-toa", filemode="w+", level=logging.DEBUG)
    if config_file is not None:
        config = yaml.load(config_file)
    else:
        config = generate_default_toa_config()
    ctx.obj = {'verbosity': verbose, 'logfile': Path(logfile),
               'loglevel': loglevel, 'config': config}


@cli.command('plot-timing')
@click.option('-o', '--output-file', type=click.Path(dir_okay=False),
              help='file to write the plot to')
def plot_timing():
    """
    plot a diagram showing the internal timing of the two stage delay line tdc
    """
    logger = logging.getLogger('plot-timing')
    toa = ToA(
        # frequency of the clock that gives the edges to the counter
        clock_frequency=160_000_000,
        # rms of the jitter of the clock in picoseconds
        clock_jitter_rms=5.,
        # amount of buffers in the ctdc delay line
        ctdc_buffer_count=38,
        # nominal delay time of a single buffer
        ctdc_delay_time=196.,
        ctdc_delay_time_rms=2.,                   # rms of the delay time of a buffer
        # number of buffers in the ftdc delay line
        ftdc_buffer_count=36,
        ftdc_delay_time=196.,                     # nominal delay of the ftdc buffers
        ftdc_delay_time_rms=2.,                   # rms of the ftdc buffer delay time
        # rms of the mismatch of the delay time of the for the start and
        # stop signal
        rgen_delay_mismatch=2.,
        # max amplification factor of the amplifier
        amp_max_ampfactor=8,
        # rms of the signal distortion factor from one buffer to the other in
        # the amplifier delay line
        amp_buffer_distortion_factor_rms=0.01,
        # rms of the signal distortion caused by the multi input or gate that
        # generates the pulse train
        amp_or_gate_distortion_factor_rms=0.01,
        # 'period' of the pulse train
        amp_max_signal_time=800.,
        ctdc_sig_ref_weight=0.2,
        ftdc_sig_ref_weight=0.2
    )
    toa.t_amp.amplification_gain_code = 3
    toa.ctdc.ref = 31
    codes = []
    times = []
    for event_t in range(0, 300, 3):
        codes.append(toa.convert(event_t))
        print(event_t, codes[-1])
        times.append(event_t)
        logging.debug("")
    plt.plot(times, codes)
    plt.xlabel('TDC Code')
    plt.ylabel('count')
    plt.savefig('toa-scan.pdf')
