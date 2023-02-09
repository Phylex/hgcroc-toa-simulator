import pytest
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import cm
from .toa import TDC
from .toa import ToA


def test_single_stage_tc():
    """
    Test the characteristics of the single stage TDC
    """
    buffer_count = 40
    lsb_delay_time = 24.4 * (buffer_count + 1)
    nominal_max_time = buffer_count * lsb_delay_time
    delay_time = lsb_delay_time * 0.01
    tdc = TDC(name='ctdc', delay_time_rms=delay_time,
              nominal_buffer_delay_time=lsb_delay_time, buffer_count=buffer_count)
    assert len(tdc._time_bin_edges_with_factors) == buffer_count + 1
    buffer_delay_times = [tdc.time_of_activation(0)]
    buffer_delay_times += [tdc.time_of_activation(
        i) - tdc.time_of_activation(i-1) for i in range(1, buffer_count)]
    fig, axis = plt.subplots()
    axis.bar(list(map(lambda x: x + 1, range(buffer_count))), buffer_delay_times)
    axis.set_xlabel('TDC Buffer')
    axis.set_ylabel('Buffer Turn on time [ps]')
    axis.set_title('Simulated TDC buffer turn on times')
    axis.text(10, 0.2*lsb_delay_time,
              f"Nominal Buffer delay time: {lsb_delay_time:.5}",
              bbox={'facecolor': 'lightgray', 'alpha': 1, 'pad': 10})
    fig.savefig('simulated_tdc.pdf')
    plt.close(fig)

    color_map = cm.get_cmap('coolwarm')

    fig, axis = plt.subplots()
    delay_buffers = list(map(lambda x: x + 0.5, range(buffer_count + 1)))
    norm = colors.Normalize(min(delay_buffers), max(delay_buffers))
    for i in [0, 8, 16, 24, 31]:
        tdc.sig = i
        buffer_delay_times = [tdc.time_of_activation(0)]
        buffer_delay_times += [tdc.time_of_activation(
            i) - tdc.time_of_activation(i-1) for i in range(1, buffer_count)]
        axis.hist(delay_buffers[:-1],
                  bins=delay_buffers,
                  weights=buffer_delay_times,
                  color=color_map(norm(i)),
                  label=f'SIG = {i}',
                  stacked=False,
                  fill=False,
                  histtype='step')
    fig.legend()
    axis.set_xlabel('TDC Buffer')
    axis.set_ylabel('Buffer Turn on time [ps]')
    axis.set_title('Simulated TDC buffer turn on times')
    axis.text(10, 0.2*lsb_delay_time,
              f"Nominal Buffer delay time: {lsb_delay_time:.5}",
              bbox={'facecolor': 'lightgray', 'alpha': .5, 'pad': 10})
    fig.savefig('simulated_tdc_sig_scan.pdf')
    plt.close(fig)

    tdc.sig = 0

    fig, axis = plt.subplots()
    delay_buffers = list(map(lambda x: x + 0.5, range(buffer_count + 1)))
    norm = colors.Normalize(min(delay_buffers), max(delay_buffers))
    for i in [0, 8, 16, 24, 31]:
        tdc.ref = i
        buffer_delay_times = [tdc.time_of_activation(0)]
        buffer_delay_times += [tdc.time_of_activation(
            i) - tdc.time_of_activation(i-1) for i in range(1, buffer_count)]
        axis.hist(delay_buffers[:-1],
                  bins=delay_buffers,
                  weights=buffer_delay_times,
                  color=color_map(norm(i)),
                  label=f'REF = {i}',
                  stacked=False,
                  fill=False,
                  histtype='step')
    fig.legend()
    axis.set_xlabel('TDC Buffer')
    axis.set_ylabel('Buffer Turn on time [ps]')
    axis.set_title('Simulated TDC buffer turn on times')
    axis.text(10, 0.2*lsb_delay_time,
              f"Nominal Buffer delay time: {lsb_delay_time:.5}",
              bbox={'facecolor': 'lightgray', 'alpha': .5, 'pad': 10})
    fig.savefig('simulated_tdc_ref_scan.pdf')
    plt.close(fig)

    tdc.ref = 0
    norm = colors.Normalize(0, 63)
    fig, axis = plt.subplots()
    delay_buffers = list(map(lambda x: x + 0.5, range(buffer_count)))
    norm = colors.Normalize(min(delay_buffers), max(delay_buffers))
    for i in [0, 8, 16, 24, 32, 40, 48, 56, 63]:
        tdc.chan_toa = i
        buffer_delay_times = [tdc.time_of_activation(
            i) - tdc.time_of_activation(i-1) for i in range(1, buffer_count)]
        axis.hist(delay_buffers[:-1],
                  bins=delay_buffers,
                  weights=buffer_delay_times,
                  color=color_map(norm(i)),
                  label=f'CHAN_TOA = {i}',
                  stacked=False,
                  fill=False,
                  histtype='step')
    fig.legend(loc='center')
    axis.set_xlabel('TDC Buffer')
    axis.set_ylabel('Buffer Turn on time [ps]')
    axis.set_title('Simulated TDC buffer turn on times')
    axis.text(10, 0.2*lsb_delay_time,
              f"Nominal Buffer delay time: {lsb_delay_time:.5}",
              bbox={'facecolor': 'lightgray', 'alpha': .5, 'pad': 10})
    fig.savefig('simulated_tdc_toa_scan.pdf')
    plt.close(fig)


def test_full_toa_scan():
    """
    Function to test the implementation of the ToA
    """
    toa = ToA(
        # frequency of the clock that gives the edges to the counter
        clock_frequency=160_000_000,
        # rms of the jitter of the clock in picoseconds
        clock_jitter_rms=5.,
        # amount of buffers in the ctdc delay line
        ctdc_buffer_count=40,
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
        amp_max_signal_time=800.
    )
    toa.t_amp.amplification_gain_code = 3
    codes = []
    times = []
    for event_t in range(0, 26000, 10):
        codes.append(toa.convert(event_t))
        print(event_t, codes[-1])
        times.append(event_t)
    plt.plot(times, codes)
    plt.savefig('toa-scan.pdf')
    plt.close()


def test_ref_sig_scan():
    toa = ToA(
        # frequency of the clock that gives the edges to the counter
        clock_frequency=160_000_000,
        # rms of the jitter of the clock in picoseconds
        clock_jitter_rms=5.,
        # amount of buffers in the ctdc delay line
        ctdc_buffer_count=40,
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
        ctdc_sig_ref_weight=0.3,
        ftdc_sig_ref_weight=0.3
    )
    toa.t_amp.amplification_gain_code = 3
    for sig in range(0, 32, 8):
        toa.ctdc.sig = sig
        codes = []
        times = []
        for event_t in range(0, 24500, 10):
            codes.append(toa.convert(event_t))
            times.append(event_t)
        plt.scatter(times, np.array(codes) -
                    (np.array(times) // 24.4) % 1024, label=f"sig = {sig}")
    plt.legend()
    plt.savefig('sig-scan.pdf')
    plt.close()


def test_statistical_toa_scan():
    """
    Function to test the implementation of the ToA
    """
    toa = ToA(
        # frequency of the clock that gives the edges to the counter
        clock_frequency=160_000_000,
        # rms of the jitter of the clock in picoseconds
        clock_jitter_rms=5.,
        # amount of buffers in the ctdc delay line
        ctdc_buffer_count=40,
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
        amp_max_signal_time=800.
    )
    toa.t_amp.amplification_gain_code = 3
    codes = []
    times = []
    for event_t in np.random.rand(100000) * 25000:
        codes.append(toa.convert(event_t))
        times.append(event_t)
    hist, bins, _ = plt.hist(codes, bins=range(1024))
    plt.plot(range(1023), np.cumsum(hist) / sum(hist) * 100)
    plt.savefig('toa-hist.pdf')
    plt.close()


def test_statistical_sig_scan():
    """
    Function to test the implementation of the ToA
    """
    toa = ToA(
        # frequency of the clock that gives the edges to the counter
        clock_frequency=160_000_000,
        # rms of the jitter of the clock in picoseconds
        clock_jitter_rms=5.,
        # amount of buffers in the ctdc delay line
        ctdc_buffer_count=40,
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
        amp_max_signal_time=800.
    )
    toa.t_amp.amplification_gain_code = 3
    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=0, vmax=32)
    sigs = [0, 8, 16, 24, 31]
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), layout='constrained')
    for sig, ax in zip(sigs, axes[::-1]):
        toa.ctdc.sig = sig
        code = []
        time = []
        for event_t in np.random.rand(100000) * 25000:
            code.append(toa.convert(event_t))
            time.append(event_t)
        hist, _, _ = ax.hist(code,
                             bins=np.arange(1024) - 0.5,
                             label=f"SIG = {sig}",
                             color=cmap(norm(sig)))
        # ax.set_xlabel('TDC code')
        ax.sharex(axes[0])
        ax.set_ylabel('Code occupancy')
        ax.legend()
    axes[0].set_title(
        'Simulated TDC-code occupancy for 100000 uniformly distributed triggers')
    axes[-1].set_xlabel('TDC Code')
    fig.savefig(f"toa-hist-sig-scan.pdf")
    plt.close(fig)


def test_statistical_ref_scan():
    """
    Function to test the implementation of the ToA
    """
    toa = ToA(
        # frequency of the clock that gives the edges to the counter
        clock_frequency=160_000_000,
        # rms of the jitter of the clock in picoseconds
        clock_jitter_rms=5.,
        # amount of buffers in the ctdc delay line
        ctdc_buffer_count=40,
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
        amp_max_signal_time=800.
    )
    toa.t_amp.amplification_gain_code = 3
    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=0, vmax=32)
    refs = [0, 8, 16, 24, 31]
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), layout='constrained')
    for ref, ax in zip(refs, axes[::-1]):
        toa.ctdc.ref = ref
        code = []
        time = []
        for event_t in np.random.rand(100000) * 25000:
            code.append(toa.convert(event_t))
            time.append(event_t)
        hist, _, _ = ax.hist(code,
                             bins=np.arange(1024) - 0.5,
                             label=f"REF = {ref}",
                             color=cmap(norm(ref)))
        # ax.set_xlabel('TDC code')
        ax.sharex(axes[0])
        ax.set_ylabel('Code occupancy')
        ax.legend()
    axes[0].set_title(
        'Simulated TDC-code occupancy for 100000 uniformly distributed triggers')
    axes[-1].set_xlabel('TDC Code')
    fig.savefig(f"toa-hist-ref-scan.pdf")
    plt.close(fig)


if __name__ == "__main__":
    test_single_stage_tc()
    test_ref_sig_scan()
    test_statistical_toa_scan()
    test_statistical_sig_scan()
    test_statistical_ref_scan()
