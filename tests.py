import pytest
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import cm
from toa import TDC


def test_single_stage_tc():
    """
    Test the characteristics of the single stage TDC
    """
    buffer_count = 32
    lsb_delay_time = 24.4 * (buffer_count + 1)
    nominal_max_time = buffer_count * lsb_delay_time
    delay_time = lsb_delay_time * 0.01
    tdc = TDC(name='ctdc', delay_time=delay_time,
              nominal_buffer_delay_time=lsb_delay_time, buffer_count=buffer_count)
    assert len(tdc._time_bin_edges_with_factors) == buffer_count + 1
    buffer_delay_times = [tdc.get_buffer_delay(i) for i in range(buffer_count)]
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
        tdc.set_sig(i)
        axis.hist(delay_buffers[:-1], bins=delay_buffers, weights=[
                  tdc.get_buffer_delay(i) for i in range(buffer_count)],
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

    tdc.set_sig(0)

    fig, axis = plt.subplots()
    delay_buffers = list(map(lambda x: x + 0.5, range(buffer_count + 1)))
    norm = colors.Normalize(min(delay_buffers), max(delay_buffers))
    for i in [0, 8, 16, 24, 31]:
        tdc.set_ref(i)
        axis.hist(delay_buffers[:-1], bins=delay_buffers, weights=[
                  tdc.get_buffer_delay(i) for i in range(buffer_count)],
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

    tdc.set_ref(0)
    norm = colors.Normalize(0, 63)
    fig, axis = plt.subplots()
    delay_buffers = list(map(lambda x: x + 0.5, range(buffer_count + 1)))
    norm = colors.Normalize(min(delay_buffers), max(delay_buffers))
    for i in [0, 8, 16, 24, 32, 40, 48, 56, 63]:
        tdc.set_chan_toa(i)
        axis.hist(delay_buffers[:-1], bins=delay_buffers, weights=[
                  tdc.get_buffer_delay(i) for i in range(buffer_count)],
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


if __name__ == "__main__":
    test_single_stage_tc()
