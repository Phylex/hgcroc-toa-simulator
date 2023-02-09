import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from .toa import ToA


# use qt backend
matplotlib.use("qtagg")
matplotlib.rcParams['toolbar'] = 'None'


def plot_toa_internals(
        outputname,
        event_t,
        time_of_next_counter_edge,
        ctdc_buffer_activation_times,
        ctdc_nominal_switching_time,
        ctdc_code,
        ctdc_num,
        residue,
        amplified_residue,
        ftdc_buffer_activation_times,
        ftdc_nominal_switching_time,
        ftdc_code,
        ftdc_num,
        amplification_gain_code,
        counter_val,
        sig,
        ref,
        channel_wise_trim_ctdc,
        fsig,
        fref,
        channel_wise_trim_ftdc,
        show=False):
    # center height of the different rows of the plot
    prc = [0.5, 2, 3.5, 5, 6.5]
    x_max = max(ctdc_buffer_activation_times[-1] + event_t + 50,
                time_of_next_counter_edge + amplified_residue + 50)
    x_max = max(x_max, time_of_next_counter_edge +
                ftdc_buffer_activation_times[17] + 25)

    fig, ax = plt.subplots(figsize=(16, 9))
    y_tick_locations = []
    y_tick_labels = []

    # plot the row for the event_t timing
    y_tick_locations.append(prc[0])
    y_tick_labels.append('ToA-Comparator')
    x_data = [event_t - 100,
              event_t,
              event_t,
              x_max]
    y_data = list(map(lambda y: y + prc[0] - .5, [0, 0, 1, 1]))
    ax.plot(x_data, y_data, color='black')
    ax.vlines([event_t], [prc[0]+.6], [prc[1]+.5],
              linestyle='--', color='grey')
    ax.text(event_t, prc[0],
            "ToA =\n" + str(np.round(event_t, decimals=1)) + ' ps',
            va='center',
            ha='center',
            bbox={'edgecolor': 'black', 'facecolor': 'white', 'boxstyle': 'round'})

    # plot the row showing the arrival of the 160MHz clock edge
    edge_time_plot_height = [prc[-1] - .5,
                             prc[-1] - .5, prc[-1] + .5, prc[-1] + .5]
    y_tick_locations.append(prc[-1])
    y_tick_labels.append('160MHz\nclock edge')
    ax.plot([event_t - 100, time_of_next_counter_edge, time_of_next_counter_edge,
             x_max],
            edge_time_plot_height, color='black')
    ax.vlines([time_of_next_counter_edge], [prc[2] + .6],
              [prc[-1] - .6], linestyle='--', color='grey')
    ax.vlines([time_of_next_counter_edge], [prc[1] - .5],
              [prc[2]-.6], linestyle='--', color='grey')
    ax.text(time_of_next_counter_edge, prc[-1], format(counter_val >> 8, '#b'),
            va='center', ha='center',
            bbox={'edgecolor': 'black', 'facecolor': 'white', 'boxstyle': 'round'})

    # plot the lines showing the activation times of the delay line buffers
    buffer_y_dim = [prc[1] - .5, prc[1] + .5]
    y_tick_locations.append(prc[1])
    average_ctdc_delay_time = np.mean(list(map(lambda x: x[1] - x[0], zip(
        ctdc_buffer_activation_times[:-1], ctdc_buffer_activation_times[1:]))))
    y_tick_labels.append(
        'CTDC\nbuffer activation')
    buff_counter = 0
    for _, (ctdc_bat, ctdc_c) in enumerate(zip(ctdc_buffer_activation_times[1:],
                                               ctdc_code[1:])):
        FMT = '-'
        NUM = '1'
        if ctdc_c == 0:
            COLOR = 'black'
            if buff_counter >= 2:
                NUM = '0'
            buff_counter += 1
        else:
            COLOR = 'red'
        ax.plot([event_t + ctdc_bat, event_t + ctdc_bat],
                buffer_y_dim, FMT, color=COLOR)
        ax.text(event_t + ctdc_bat, prc[1], NUM, va='center', ha='center',
                bbox={'edgecolor': COLOR,
                      'facecolor': 'white',
                      'boxstyle': 'round'})

    # plot the reference marks for the delay line buffers of the ctdc
    if ctdc_nominal_switching_time is not None:
        for i, code in enumerate(ctdc_code[1:]):
            if code == 0:
                COLOR = 'black'
                if buff_counter >= 2:
                    NUM = '0'
                buff_counter += 1
            else:
                COLOR = 'red'
            ax.plot([event_t + (i + 1) * ctdc_nominal_switching_time, event_t + (i + 1) * ctdc_nominal_switching_time],
                    buffer_y_dim, '--', color=COLOR, alpha=0.3)

    # plot the residue that is generated before being sent to the time amplifier
    residue_gen_reset_time = ctdc_buffer_activation_times[ctdc_code.index(
        0) + 1]
    x_data = [event_t - 100,
              time_of_next_counter_edge,
              time_of_next_counter_edge,
              event_t + residue_gen_reset_time,
              event_t + residue_gen_reset_time,
              x_max]
    y_data = list(map(lambda y: y + prc[2] - .5, [0, 0, 1, 1, 0, 0]))
    ax.plot(x_data, y_data, color='black')
    ax.vlines([event_t + residue_gen_reset_time], [prc[1]+.5],
              [prc[2]-.6], linestyle='--', color='grey')
    y_tick_locations.append(prc[2])
    y_tick_labels.append('CTDC residue')

    # plot the amplified residue
    x_data = [event_t + residue_gen_reset_time + 20,
              time_of_next_counter_edge + amplified_residue,
              time_of_next_counter_edge + amplified_residue]
    y_data = list(map(lambda y: y + prc[2] - .5, [1, 1, .05]))
    ax.plot(x_data, y_data, color='blue')
    text_x = (time_of_next_counter_edge + amplified_residue - event_t -
              residue_gen_reset_time - 20) / 2 + event_t + residue_gen_reset_time + 20
    ax.text(text_x, prc[2] + .5,
            f'x{2**amplification_gain_code} amplified residue',
            va='center', ha='center',
            bbox={'edgecolor': 'blue', 'facecolor': 'white', 'boxstyle': 'round'})
    ax.vlines([time_of_next_counter_edge + amplified_residue], [prc[2]+.6],
              [prc[-2]+.5], linestyle='--', color='grey')

    # plot the reference marks for the delay line buffers of the ctdc
    buffer_y_dim = [prc[-2] - .5, prc[-2] + .5]
    if ftdc_nominal_switching_time is not None:
        for i, code in enumerate(ftdc_code[1:20]):
            if code == 0:
                COLOR = 'black'
                if buff_counter >= 2:
                    NUM = '0'
                buff_counter += 1
            else:
                COLOR = 'red'
            ax.plot([time_of_next_counter_edge + (i + 1) * ftdc_nominal_switching_time, time_of_next_counter_edge + (i + 1) * ftdc_nominal_switching_time],
                    buffer_y_dim, '--', color=COLOR, alpha=0.3)

    # plot the FTDC buffers
    y_tick_locations.append(prc[-2])
    y_tick_labels.append('FTDC buffer\nactiviation')
    average_ftdc_delay_time = np.mean(list(map(lambda x: x[1] - x[0], zip(
        ftdc_buffer_activation_times[:-1], ftdc_buffer_activation_times[1:]))))
    for i, (ftdc_bit, ftdc_b_delay_time) in enumerate(
            zip(ftdc_code[1:20], ftdc_buffer_activation_times[1:20])):
        FMT = '-'
        NUM = '1'
        COLOR = 'black'
        if ftdc_bit == 0:
            NUM = '0'
        else:
            COLOR = 'red'
        plot_ftdc_b_time = time_of_next_counter_edge + ftdc_b_delay_time
        ax.plot([plot_ftdc_b_time, plot_ftdc_b_time],
                buffer_y_dim, FMT, color=COLOR)
        ax.text(plot_ftdc_b_time, prc[-2], NUM, va='center', ha='center',
                bbox={'edgecolor': COLOR, 'facecolor': 'white', 'boxstyle': 'round'})
    ax.set_yticks(y_tick_locations, y_tick_labels)

    # add the text that explains the calculation of the ToA output
    ax.text((event_t + x_max) / 2,
            prc[0], f'ToA calculation\n'
            f'((counter << 8) - (CTDC-code << {amplification_gain_code}) + FTDC-code) & 0x3ff\n'
            f'= ({counter_val} - {ctdc_num} + {ftdc_num}) & 0x3ff\n'
            f'= {(counter_val - ctdc_num + ftdc_num) & 0x3ff}',
            va='center', ha='center',
            bbox={'edgecolor': 'black', 'facecolor': 'lightcyan', 'boxstyle': 'round'})
    # print the text that shows the calculation for the CTDC code
    ax.text(event_t + ctdc_buffer_activation_times[-1] / 2, prc[1]-.7,
            r'CTDC-code = sum(CTDC-buffer-state) + 2 = ' + f'{ctdc_num >> 3}',
            va='center', ha='center',
            bbox={'edgecolor': 'black', 'facecolor': 'white', 'boxstyle': 'round'})
    ax.text(time_of_next_counter_edge + ftdc_buffer_activation_times[18] / 2, prc[-2]+.7,
            r'FTDC-code = sum(CTDC-buffer-state) = ' + f'{ftdc_num}',
            va='center', ha='center',
            bbox={'edgecolor': 'black', 'facecolor': 'white', 'boxstyle': 'round'})
    ax.annotate(f'CTDC-buffer\naverage delay time\n'
                f'= {average_ctdc_delay_time:.1f} ps\n\nFTDC-buffer\n'
                f'average delay time\n= {average_ftdc_delay_time:.1f} ps\n\n'
                f'CTRL_IN_SIG_CTDC={sig}\nCTRL_IN_REF_CTDC={ref}\n'
                f'CTRL_IN_SIG_FTDC={fsig}\nCTRL_IN_REF_FTDC={fref}',
                xy=(1, 1), xycoords='axes fraction',
                xytext=(30, -40), textcoords='offset points',
                ha="center", va="top",
                bbox={'edgecolor': 'black', 'facecolor': 'white', 'boxstyle': 'round'})
    ax.set_xlabel('Time [ps]')
    ax.set_title('ToA internal timing and code generation')
    fig.savefig(outputname)
    if show:
        plt.show()
    plt.close(fig)


# test the plot
if __name__ == "__main__":
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
        amp_max_signal_time=800.
    )
    toa.t_amp.amplification_gain_code = 3
    toa.ctdc.ref = 30
    event_t = 20
    (time_of_next_counter_edge, ctdc_buffer_activation_times, ctdc_code, residue, amplified_residue,
     ftdc_buffer_activation_times, ftdc_code, ftdc_num,
     ctdc_num, counter_val) = toa.convert(event_t, plot_data=True)
    plot_toa_internals('toa-timing.pdf',
                       ctdc_nominal_switching_time=196.,
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
                       amplification_gain_code=toa.t_amp._amplification_gain_code,
                       ftdc_buffer_activation_times=ftdc_buffer_activation_times,
                       ftdc_code=ftdc_code,
                       ftdc_num=ftdc_num,
                       counter_val=counter_val,
                       show=True)
