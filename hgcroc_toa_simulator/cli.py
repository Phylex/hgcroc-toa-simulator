"""
Module containing the main command line applications that
are available to the user
"""
import logging
from pathlib import Path
from . import __version__
import click
import yaml
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
              type=click.Choice(log_level_dict.keys(),
                                case_sensitive=False),
              default='INFO',
              help='Set the loglevel for the log file')
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
        ctdc_sig, ctdc_ref,
        ftdc_sig, ftdc_ref, amplification_gain_code,
        ctdc_delay_time, ctdc_buffer_count, ftdc_delay_time,
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
    tdc = ToA(
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
              help="Set the ctdc sig parameters that are to be plotted")
@click.option("-fs", "--fsig", type=click.IntRange(0, 31), multiple=True,
              default=[0],
              show_default=True,
              help="Set the ftdc sig parameters that are to be plotted")
@click.option("-cr", "--ref", type=click.IntRange(0, 31), multiple=True,
              default=[0],
              show_default=True,
              help="Set the REF parameters of the CTDC that are to be plotted")
@click.option("-fr", "--fref", type=click.IntRange(0, 31), multiple=True,
              default=[0],
              show_default=True,
              help="Set the REF parameters of the FTDC that are to be plotted")
@click.option('-c', "--colormap", type=str,
              default='blue',
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
    cmap = mpl.colormaps[colormap]
    norm = mpl.colors.Normalize(vmin=min(-ref_vals), vmax=max(sig_vals))
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
