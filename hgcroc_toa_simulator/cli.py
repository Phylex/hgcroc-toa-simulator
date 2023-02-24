"""
Module containing the main command line applications that
are available to the user
"""
from typing import Union
import logging
import yaml
from pathlib import Path
from . import __version__
import sys
import click
from .utils import set_up_logger
import numpy as np
from .toa import ToA
from .plot_toa_timing_test import plot_toa_internals
import matplotlib.pyplot as plt
import matplotlib as mpl

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
              type=click.Choice([str(k) for k in log_level_dict.keys()],
                                case_sensitive=False),
              default='INFO',
              help='Set the loglevel for the log file')
@click.option('-cf', '--config-file', type=click.Path(dir_okay=False,
                                                      file_okay=True),
              default=None,
              help="path to the configuration file containing the exact "
                   "settings for the TDC for reproducable results accross "
                   "invocations. If this option is set the options used to "
                   "generate the TDC parameters are ignored, the trimming "
                   "parameters are still applied")
@click.option('-cs', '--ctdc-sig', type=click.IntRange(0, 31),
              default=0, help='Sets CTRL_IN_SIG_CTDC_P_D of the simulated TDC')
@click.option('-cr', '--ctdc-ref', type=click.IntRange(0, 31),
              default=0, help='Sets CTRL_IN_REF_CTDC_P_D of the simulated TDC')
@click.option('-fs', '--ftdc-sig', type=click.IntRange(0, 31),
              default=0, help='Sets CTRL_IN_SIG_FTDC_P_D of the simulated TDC')
@click.option('-fr', '--ftdc-ref', type=click.IntRange(0, 31),
              default=0, help='Sets CTRL_IN_REF_FTDC_P_D of the simulated TDC')
@click.option('-a', "--amplification-gain-code", type=click.IntRange(0, 3),
              default=3, help="Amplication gain code, "
              "3 := 8x, 2 := 4x, 1 := 2x, 0 := 1x")
@click.option('-cd', "--ctdc-delay-time",
              type=click.FloatRange(0, max_open=True), default=196.,
              show_default=True,
              help="Nominal delay of the CTDC delay time buffer")
@click.option('-cc', "--ctdc-buffer-count",
              type=click.IntRange(2, max_open=True), default=40,
              show_default=True,
              help="Number of delay line buffers in the CTDC")
@click.option('-fd', "--ftdc-delay-time",
              type=click.FloatRange(0, max_open=True), default=196.,
              show_default=True,
              help="Nominal delay of the FTDC delay time buffer")
@click.option('-fc', "--ftdc-buffer-count",
              type=click.IntRange(2, max_open=True), default=20,
              show_default=True,
              help="Number of delay line buffers in the FTDC")
@click.option("-clkf", "--clock-frequency",
              type=click.FloatRange(1, max_open=True),
              default=160_000_000., show_default=True,
              help="Clock frequency of the clock driving the delay line stop "
              "signal and counter")
@click.option("-clkj", "--clock-jitter-rms",
              type=click.FloatRange(0, max_open=True),
              default=5, show_default=True,
              help="The rms of the jitter of the clock in picoseconds")
@click.option("-mr", "--max_residue",
              type=click.FloatRange(0, 2000), default=800.,
              help="The maximum residue that the time amplifier can amplify")
@click.version_option(__version__)
@click.pass_context
def cli(ctx, verbose,
        logfile: click.Path,
        loglevel: int,
        config_file: Union[str, None],
        ctdc_sig: int, ctdc_ref: int,
        ftdc_sig: int, ftdc_ref: int,
        amplification_gain_code: int,
        ctdc_delay_time: float,
        ctdc_buffer_count,
        ftdc_delay_time,
        ftdc_buffer_count,
        clock_frequency, clock_jitter_rms, max_residue):
    """
    Simulate and explore the ToA as implemented in the HGCROCv3
    """
    # in this function we can do all the things that would need doing
    # for every invocation of the tool, but nothing other than delegating
    # the actual tasks to commands
    # logging.basicConfig(filename="log-toa", filemode="w+", level=logging.DEBUG)
    logger = logging.getLogger('toa-simulator')
    set_up_logger(logger, verbosity_level=verbose,
                  logfile=Path(logfile), loglevel=loglevel)
    if config_file is not None:
        tdc = ToA.from_config_file(config_file)
    else:
        tdc = ToA.from_parameters(
            # frequency of the clock that gives the edges to the counter
            clock_frequency=clock_frequency,
            # rms of the jitter of the clock in picoseconds
            clock_jitter_rms=clock_jitter_rms,
            # amount of buffers in the ctdc delay line
            ctdc_buffer_count=ctdc_buffer_count,
            # nominal delay time of a single buffer
            ctdc_delay_time=ctdc_delay_time,
            ctdc_delay_time_rms=2.,  # rms of the delay time of a buffer
            # number of buffers in the ftdc delay line
            ftdc_buffer_count=ftdc_buffer_count,
            # nominal delay of the ftdc buffers
            ftdc_delay_time=ftdc_delay_time,
            ftdc_delay_time_rms=2.,  # rms of the ftdc buffer delay time
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
            amp_max_signal_time=max_residue,
            ctdc_sig_ref_weight=0.2,
            ftdc_sig_ref_weight=0.2
        )
    tdc.t_amp.amplification_gain_code = amplification_gain_code
    tdc.ctdc.sig = ctdc_sig
    tdc.ctdc.ref = ctdc_ref
    tdc.ftdc.sig = ftdc_sig
    tdc.ftdc.ref = ftdc_ref
    ctx.obj = {'logger': logger,
               'tdc': tdc}


@cli.command('plot-timing')
@click.argument("time-of-arrival", type=click.FloatRange(0, 25000))
@click.option('-o', '--output-file',
              type=click.Path(dir_okay=False),
              default='toa-timing.pdf',
              help='file to write the plot to')
@click.pass_context
def plot_timing(ctx, time_of_arrival, output_file):
    """
    plot a diagram showing the internal timing of the two stage delay line tdc
    """
    toa = ctx.obj['tdc']
    (time_of_next_counter_edge, ctdc_buffer_activation_times, ctdc_code,
     residue, amplified_residue, ftdc_buffer_activation_times,
     ftdc_code, ftdc_num, ctdc_num,
     counter_val) = toa.convert(time_of_arrival=time_of_arrival,
                                plot_data=True)
    plot_toa_internals(
        outputname=output_file,
        event_t=time_of_arrival,
        ctdc_nominal_switching_time=toa.nominal_ctdc_delay_time,
        time_of_next_counter_edge=time_of_next_counter_edge,
        sig=toa.ctdc.sig,
        fsig=toa.ftdc.sig,
        channel_wise_trim_ctdc=toa.ctdc.chan_toa,
        ref=toa.ctdc.ref,
        fref=toa.ftdc.ref,
        channel_wise_trim_ftdc=toa.ftdc.chan_toa,
        ctdc_buffer_activation_times=ctdc_buffer_activation_times,
        ctdc_code=ctdc_code,
        ctdc_num=ctdc_num,
        residue=residue,
        amplified_residue=amplified_residue,
        amplification_gain_code=toa.t_amp.amplification_gain_code,
        ftdc_buffer_activation_times=ftdc_buffer_activation_times,
        ftdc_nominal_switching_time=toa.nominal_ftdc_delay_time,
        ftdc_code=ftdc_code,
        ftdc_num=ftdc_num,
        counter_val=counter_val,
        show=True)


@cli.command("create-config")
@click.argument("config-file", type=click.File(mode='w+', encoding='utf-8'))
@click.pass_context
def create_config(ctx, config_file):
    config_file.write(yaml.dump(ctx.obj["tdc"].export_config()))


@cli.command('histogram')
@click.argument("events", type=click.IntRange(0, max_open=True))
@click.option("-t", "--type", "tdc_type",
              type=click.Choice(["ctdc", "ftdc", "toa"],
                                case_sensitive=False),
              help="determin the result plotted. toa will result in the entire ToA range being generated")
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Store the resulting plot at <output>")
@click.option("-sh", "--show", is_flag=True, default=False,
              help="Show plot interactively")
@click.option("-cs", "--sig", type=click.IntRange(0, 31), multiple=True,
              default=[0],
              show_default=True,
              help="create one step histogram for each value of SIG provided here")
@click.option("-fs", "--fsig", type=click.IntRange(0, 31), multiple=True,
              default=[0],
              show_default=True,
              help="create one step histogram for each value of FTDC-SIG provided here")
@click.option("-cr", "--ref", type=click.IntRange(0, 31), multiple=True,
              default=[0],
              show_default=True,
              help="create one step histogram for each value of REF provided here")
@click.option("-fr", "--fref", type=click.IntRange(0, 31), multiple=True,
              default=[0],
              show_default=True,
              help="create one step histogram for each value of FTDC-REF provided here")
@click.option('-c', "--colormap", type=str,
              default='viridis',
              help="color of the histogram")
@click.option("-w", "--width", type=click.IntRange(0, max_open=True), default=8,
              help="Width of the figure")
@click.option("-h", "--height", type=click.IntRange(0, max_open=True), default=6,
              help="height of the figure")
@click.option("-m", "--mod-visible", is_flag=True, default=False,
              help="The bin is going to be mod(width) where width "
                   "is the amount of bits of the parameter")
@click.pass_context
def histogram(ctx, tdc_type, events, output, show, sig,
              fsig, ref, fref, colormap, width, height, mod_visible):
    toa = ctx.obj['tdc']
    fig, ax = plt.subplots(figsize=(width, height), layout='constrained')
    if (len(sig) > 1 or len(ref) > 1) and (len(fsig) > 1 or len(fref) > 1):
        click.echo("scanning both sig/ref and fsig/fref not supported")
        sys.exit()

    # build the lists of sig and ref parameters for whitch to create an entry
    # in the plot.
    if len(sig) > 1 or len(ref) > 1:
        sig_vals = np.array([0 for _ in ref] + list(sig))
        ref_vals = np.array(list(ref) + [0 for _ in sig])
        toa.ftdc.ref = fref[0]
        toa.ftdc.sig = fsig[0]
        trim_tdc = toa.ctdc
        scan_tdc = "CTDC"
    elif len(fref) > 1 or len(fsig) > 1:
        sig_vals = np.array([0 for _ in fref] + list(fsig))
        ref_vals = np.array(list(fref) + [0 for _ in fsig])
        scan_tdc = "FTDC"
        toa.ctdc.ref = ref[0]
        toa.ctdc.sig = sig[0]
        trim_tdc = toa.ftdc
    else:
        sig_vals = np.array([0])
        ref_vals = np.array([0])
        trim_tdc = toa.ctdc
        scan_tdc = "CTDC"

    # determin colorsceme and normalization
    cmap = mpl.colormaps[colormap]
    norm = mpl.colors.Normalize(vmin=min(-ref_vals), vmax=max(sig_vals))
    
    # calculate the bins of the histogram
    if tdc_type == "ctdc":
        if mod_visible:
            mod_base = 32
        else:
            mod_base = toa.ctdc._buffer_count
    elif tdc_type == "ftdc":
        if mod_visible:
            mod_base = 8
        else:
            mod_base = toa.ftdc._buffer_count
    else:
        mod_base = 1024
    bins = np.arange(mod_base + 1) - .5

    # create one step line in the plot for every sig-ref pair found in the
    # input data
    for sig, ref in zip(sig_vals, ref_vals):
        trim_tdc.ref = ref
        trim_tdc.sig = sig
        code = [toa.convert(e, code_type=tdc_type) % mod_base
                for e in np.random.rand(events) * 25000]
        ax.hist(code,
                bins=bins,
                label=f"{scan_tdc}-SIG: {trim_tdc.sig}, {scan_tdc}-REF: {trim_tdc.ref}",
                color=cmap(norm(sig-ref)),
                histtype='step')
    ax.set_xlabel('TDC code')
    ax.set_ylabel('Code occupancy')
    ax.legend(loc='upper left')
    ax.set_title(
        f'Simulated {tdc_type.upper()}-code occupancy for {events} uniforily distributed triggers')
    ax.set_xlabel('TDC Code')
    if output is not None:
        fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


@cli.command('generate-data')
@click.argument("events", type=click.IntRange(0, max_open=True))
@click.option("-o", "--output",
              type=click.Path(dir_okay=False),
              default=None,
              help="File to write the output to")
@click.pass_context
def generate_data(ctx, events: int, output: str) -> None:
    """
    Generate TDC codes from uniformily random input times between 0 and 25ns.

    Generates TDC codes and writes the Codes to stdout. If an output is
    provided it will write the data to the file instead of stdout. The filetype
    of the output file is a CSV with one column being the Time of arrival in
    picoseconds and the other being the code generated by the TDC. The two
    columns are separated by ','. If the output is 'stdout' it will simply print
    the ToA code, and then a newline
    """
    if output is not None:
        out = open(output, 'w+')
        file = True
    else:
        out = click.get_text_stream('stdout')
        o_type = 'stdout'
        file = False
    toa = ctx.obj["tdc"]
    events = np.random.rand(events) * 25000
    codes = [toa.convert(e, code_type="toa") for e in events]
    if file:
        lines = [f'{t},{c}\n' for t, c in zip(events, codes)]
        out.write("Time[ps],ToA-Code\n")
    else:
        lines = [f'{c}\n' for c in codes]
    out.writelines(lines)
    out.close()
