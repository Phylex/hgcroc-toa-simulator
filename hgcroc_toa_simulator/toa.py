from typing import Union
from functools import reduce
from copy import copy
from operator import mul
import logging
import matplotlib.pyplot as plt
import numpy as np


class Counter:
    """
    160 MHz counter used to generate the top two bits of the ToA
    """

    def __init__(self, jitter_rms_ps: float, frequency: float):
        self.logger = logging.getLogger('counter')
        self.period = 10**12 / frequency
        self.frequency = frequency
        self.jitter_rms = jitter_rms_ps

    def convert(self, event_time: float):
        """
        convert a timestamo into the counter code (respecting jitter)
        and provides the time of the next clock edge such as to avoid glitches
        """
        last_period_length = np.random.normal(
            loc=self.period, scale=self.jitter_rms)
        counter = np.floor(event_time / self.period)
        if event_time % self.period >= last_period_length:
            counter += 1
            next_edge = np.random.normal(
                loc=(counter + 1) * self.period, scale=self.jitter_rms)
        else:
            next_edge = (counter * self.period) + last_period_length
        return next_edge, int(counter % 256)


class TDC:
    """
    Class representing the TDC circuit all numbers here refer to picoseconds.

    The TDC class is responsible for imitating both the FTDC and CTDC behaviour.
    """

    def __init__(self, name: str,
                 delay_time_rms: Union[float, list],
                 buffer_count: int = 7,
                 nominal_buffer_delay_time: float = 195.2,
                 max_ref_sig_impact: float = .1,
                 max_chan_wise_impact: float = .1):
        self._name = name
        self._buffer_count = buffer_count
        if isinstance(delay_time_rms, list):
            if delay_time_rms[0] == 0:
                self._buffer_delay_times = delay_time_rms
            else:
                raise ValueError(
                    "The 0th index must be 0 as the first 1 switches on immediately")
        else:
            self._buffer_delay_times = [np.random.normal(loc=nominal_buffer_delay_time,
                                                         scale=delay_time_rms)
                                        for _ in range(buffer_count)]
            self._buffer_delay_times.insert(0, 0.)
        self._ref = 0
        self._max_ref_sig_factor = max_ref_sig_impact
        self._sig = 0
        self._chan_toa = 0
        self._max_chan_toa_factor = max_chan_wise_impact
        self._time_bin_edges = np.cumsum(self._buffer_delay_times)
        self._time_bin_edges_with_factors = self._time_bin_edges
        self._sig_ref_buffer_switching_time_factor = 1.
        self._channel_private_switching_time_factor = 1.
        self._set_chan_toa(0)
        self.logger = logging.getLogger(f'{self._name}')

    def _set_sig(self, sig: int) -> None:
        """
        Set the SIG parameter of the TDC. The sig parameter slows down the TDC buffers
        so that it increases the time between the input going high and the output of the
        buffer going high
        """
        if self._ref != 0 and sig != 0:
            raise ValueError(
                "The REF parameter of the %s TDC is not 0. Set REF to 0 "
                "before setting the SIG parameter != 0")
        if 0 <= sig <= 31:
            self._sig = sig
        else:
            raise ValueError(
                " SIG outside of range [0, 31]"
            )
        self._sig_ref_buffer_switching_time_factor = 1 + (
            self._sig / 32 * self._max_ref_sig_factor)
        self._time_bin_edges_with_factors = self._time_bin_edges * \
            (self._sig_ref_buffer_switching_time_factor +
             self._channel_private_switching_time_factor) / 2

    def _get_sig(self) -> int:
        return self._sig

    def _del_sig(self) -> None:
        self._sig_ref_buffer_switching_time_factor = 1
        del self._sig

    sig = property(fset=_set_sig,
                   fget=_get_sig,
                   fdel=_del_sig,
                   doc="SIG tuning parameter of the ToA-TDC, increases the "
                       "delay time of the buffers")

    def _set_ref(self, ref: int) -> None:
        """
        Set the REF parameter of the TDC. The ref parameter speeds up the TDC buffers
        so that it decreases the time between the input and the ouptut of the buffer going high.
        """
        if self._sig != 0 and ref != 0:
            raise ValueError(
                "The SIG parameter of the %s TDC is not 0. Set SIG to 0 "
                "before setting the REF parameter != 0")
        if 0 <= ref <= 31:
            self._ref = ref
        else:
            raise ValueError(
                " REF outside of range [0, 31]"
            )
        self._sig_ref_buffer_switching_time_factor = 1 - (
            self._ref / 32 * self._max_ref_sig_factor)
        self._time_bin_edges_with_factors = self._time_bin_edges * \
            (self._sig_ref_buffer_switching_time_factor +
             self._channel_private_switching_time_factor) / 2

    def _get_ref(self) -> int:
        return self._ref

    def _del_ref(self) -> None:
        self._sig_ref_buffer_switching_time_factor = 1
        del self._ref

    ref = property(fset=_set_ref,
                   fget=_get_ref,
                   fdel=_del_ref,
                   doc="REF tuning parameter of the TOA-TDC, decreases the "
                       "delay time of the buffers")

    def _set_chan_toa(self, chan_toa: int) -> None:
        if 0 <= chan_toa <= 63:
            self._chan_toa = chan_toa
        else:
            raise ValueError(
                " CHAN_TOA  outside of range [0, 63]"
            )
        self._channel_private_switching_time_factor = 1 + ((
            self._chan_toa - 32) / 32 * self._max_chan_toa_factor)
        self._time_bin_edges_with_factors = self._time_bin_edges * \
            (self._sig_ref_buffer_switching_time_factor +
             self._channel_private_switching_time_factor) / 2

    def _get_chan_toa(self) -> int:
        return self._chan_toa

    def _del_chan_toa(self) -> None:
        self._channel_private_switching_time_factor = 1
        del self._chan_toa

    chan_toa = property(fset=_set_chan_toa,
                        fget=_get_chan_toa,
                        fdel=_del_chan_toa,
                        doc="Parameter to change the buffer delay time of "
                            "the TDC buffer chain, neutral point is 31")

    def convert(self, start_time: float, stop_time: float, delta_t_uncertainty: float = 0) -> int:
        """
        Calculate the thermometer code produced by the delay line TDC.
        This function does not perform any encoding of the value
        """
        self.logger.debug(
            f"starting conversion with start_time = {start_time} "
            f"and stop_time = {stop_time}")
        delta_t = stop_time - start_time
        self.logger.debug(f"time delta = {delta_t}")
        assert delta_t >= 0
        delta_t += np.random.normal(loc=0, scale=delta_t_uncertainty)
        self.logger.debug(f"time delta after uncertainty addition: {delta_t}")
        tdc_code = list(
            map(lambda be: 1 if be <= delta_t else 0, self._time_bin_edges_with_factors))
        self.logger.debug(f"Resulting thermometer code: {tdc_code}")
        return tdc_code

    def time_of_activation(self, buffer_output_index: int) -> float:
        """
        Function to retreive the activation time of the buffer output with the
        given index
        """
        assert buffer_output_index >= 0
        if buffer_output_index >= self._buffer_count:
            raise IndexError(f"The TDC only has {self._buffer_count} "
                             "delay buffers in the chain")
        self.logger.debug(
            f"Activation time for buffer {buffer_output_index} = {self._time_bin_edges_with_factors[buffer_output_index]}")
        return self._time_bin_edges_with_factors[buffer_output_index]

    @staticmethod
    def encode(tdc_thermometer_code: list[int]):
        """
        Convert the list of buffer output states into a number that can be used by the
        calculation logic for the final ToA/ToT values
        """
        return sum(tdc_thermometer_code) - 1


class Rgen:
    """
    Class representing the residue generator
    """

    def __init__(self, delay_mismatch: Union[float, list], ctdc_buffer_count: int):
        """
        Initialise the residue generator. The important thing is to generate the
        mismatch between the two buffers that introduce the \tau_R into the system

        :param delay_mismatch: The mismatch between the Tau_R buffers from the
            stop-signal path and the TDC stop signal. This is per buffer parir, so
            there needs to be a total of CTDC buffer_count + 1 pairs
        :type delay_mismatch: Union[float, list]
        :param ctdc_buffer_count: The number of buffers of the CTDC for which to generate
            the delay
        :type ctdc_buffer_count: int
        """
        if isinstance(delay_mismatch, list):
            self._delay_mismatch = delay_mismatch
        else:
            self._delay_mismatch = np.array(
                [np.random.normal(scale=delay_mismatch)
                 for _ in range(ctdc_buffer_count - 2)])
        self.logger = logging.getLogger("Rgen")

    def generate_residue(self, ctdc: TDC,
                         ctdc_code: list[int],
                         rgen_set_time: float,
                         reset_override_time: float):
        """
        Calculate the residue of the CTDC
        """
        start_out_buffer_output_index = sum(ctdc_code)
        self.logger.debug(
            f"generating residue between t={rgen_set_time} and ctdc buffer "
            f"{start_out_buffer_output_index + 1}")
        if start_out_buffer_output_index >= len(self._delay_mismatch):
            rgen_reset_time = reset_override_time
        else:
            rgen_reset_time = ctdc.time_of_activation(
                start_out_buffer_output_index + 1) + self._delay_mismatch[start_out_buffer_output_index]
        self.logger.debug(
            f"stop time: {rgen_reset_time} for a residue of {rgen_reset_time - rgen_set_time}")
        return rgen_reset_time - rgen_set_time


class TimeAmplifier:
    """
    Class representing the time amplifier of the TDC block
    """

    def __init__(self,  max_amplification_gain: int,
                 signal_distortion_factor_rms: float, or_gate_signal_distortion_factor_rms: float,
                 max_signal_time: float, amplification_gain_code: int = 0):
        assert max_amplification_gain in [2**n for n in range(10)]
        assert 2 ** amplification_gain_code <= max_amplification_gain
        self._max_amplification_gain = max_amplification_gain
        self._amplification_gain_code = amplification_gain_code
        self._amplification_gain = 2 ** amplification_gain_code
        self.amplification_buffers_distortion = [np.random.normal(
            scale=signal_distortion_factor_rms, loc=1) for _ in range(max_amplification_gain - 1)]
        self.or_gate_distortion = [np.random.normal(
            scale=or_gate_signal_distortion_factor_rms, loc=1) for _ in range(max_amplification_gain)]
        self.max_signal_time = max_signal_time
        self.logger = logging.getLogger('TimeAmp')

    def _set_amplification_gain(self, code: int):
        assert code >= 0
        assert 2 ** code <= self._max_amplification_gain
        self._amplification_gain = 2 ** code
        self._amplification_gain_code = code

    def _get_amplification_gain(self):
        return copy(self._amplification_gain_code)

    amplification_gain_code = property(fset=_set_amplification_gain,
                                       fget=_get_amplification_gain,
                                       doc="code that sets the amplification gain. Gain = 2^code")

    def amplify(self, residue: float):
        """
        Calculates the sum of the on-time of all pulses after amplification.

        Calculate the total 'on-time' of the pulse train taking into account the distortion
        in the buffer chain that performs the pulse replication and the distortion caused by
        the OR gate that produces the pulse train from the pulses generated by
        the amplification DLL.
        """
        self.logger.debug(
            f"Amplifying a residue of {residue} by a nominal factor of {self._amplification_gain}:"
            f"total nominal residue {residue * self._amplification_gain}")
        if residue > self.max_signal_time:
            residue = self.max_signal_time
            self.logger.debug(
                f"residue larger than max signal time ({self.max_signal_time}). Limiting residue to max signal time")
        sig_dist = copy(self.amplification_buffers_distortion)
        sig_dist.insert(0, residue)
        pulses = [reduce(mul, sig_dist[:i+1])
                  for i in range(self._amplification_gain)]
        self.logger.debug(
            f"created pulse train with total time of {sum(pulses)}")
        total_amp = sum([self.or_gate_distortion[i] *
                        pulse for i, pulse in enumerate(pulses)])
        self.logger.debug(f"Generated amplified residue of: {total_amp}")
        return total_amp


class ToA:
    def __init__(self,
                 clock_frequency: float,
                 clock_jitter_rms: float,
                 ctdc_buffer_count: int,
                 ctdc_delay_time: float,
                 ctdc_delay_time_rms: float,
                 ftdc_buffer_count: int,
                 ftdc_delay_time: float,
                 ftdc_delay_time_rms: float,
                 rgen_delay_mismatch: float,
                 amp_max_ampfactor: int,
                 amp_max_signal_time: float,
                 amp_buffer_distortion_factor_rms: float,
                 amp_or_gate_distortion_factor_rms: float,
                 ctdc_sig_ref_weight: float = 0.1,
                 ctdc_channel_trim_weight: float = 0.1,
                 ftdc_sig_ref_weight: float = 0.1,
                 ftdc_channel_trim_weight: float = 0.1,
                 ):
        self.counter = Counter(frequency=clock_frequency,
                               jitter_rms_ps=clock_jitter_rms)
        self.ctdc = TDC(name='CTDC', delay_time_rms=ctdc_delay_time_rms,
                        buffer_count=ctdc_buffer_count,
                        nominal_buffer_delay_time=ctdc_delay_time,
                        max_chan_wise_impact=ctdc_channel_trim_weight,
                        max_ref_sig_impact=ctdc_sig_ref_weight)
        self.rgen = Rgen(delay_mismatch=rgen_delay_mismatch,
                         ctdc_buffer_count=ctdc_buffer_count)
        self.t_amp = TimeAmplifier(
            max_amplification_gain=amp_max_ampfactor,
            signal_distortion_factor_rms=amp_buffer_distortion_factor_rms,
            or_gate_signal_distortion_factor_rms=amp_or_gate_distortion_factor_rms,
            max_signal_time=amp_max_signal_time)
        self.ftdc = TDC(name='FTDC',
                        delay_time_rms=ftdc_delay_time_rms,
                        buffer_count=ftdc_buffer_count,
                        nominal_buffer_delay_time=ftdc_delay_time,
                        max_chan_wise_impact=ftdc_channel_trim_weight,
                        max_ref_sig_impact=ftdc_sig_ref_weight)
        self.ctdc.sig = 0
        self.ctdc.ref = 0
        self.ctdc.chan_toa = 31
        self.ftdc.sig = 0
        self.ftdc.ref = 0
        self.ftdc.chan_toa = 31
        self.logger = logging.getLogger('ToA')

    def convert(self, time_of_arrival: float, BX: int = 0, plot_data=False):
        time_of_next_counter_edge, counter_val = self.counter.convert(
            time_of_arrival)
        ctdc_code = self.ctdc.convert(
            start_time=time_of_arrival, stop_time=time_of_next_counter_edge)
        residue = self.rgen.generate_residue(
            self.ctdc,
            ctdc_code,
            rgen_set_time=time_of_next_counter_edge - time_of_arrival,
            reset_override_time=time_of_next_counter_edge+12500)
        assert residue >= 0
        amplified_residue = self.t_amp.amplify(residue)
        ftdc_code = self.ftdc.convert(
            start_time=0, stop_time=amplified_residue)

        # convert the TDC outputs into binary, shift them all into the
        ctdc_num = TDC.encode(ctdc_code)
        self.logger.debug(f"encoded CTDC output: {ctdc_num} = {bin(ctdc_num)}")
        ftdc_num = TDC.encode(ftdc_code)
        self.logger.debug(f"encoded FTDC output: {ftdc_num} = {bin(ftdc_num)}")
        # the two is added to the ctdc num as this is the code that the ftdc uses as the stop
        # signal for it's conversion
        ctdc_num = (ctdc_num + 2) << self.t_amp.amplification_gain_code
        self.logger.debug(f"shifted CTDC output: {ctdc_num}")
        tdc_num = ctdc_num - ftdc_num
        self.logger.debug(f"tdc_code = {tdc_num}")
        counter_val = (counter_val + 1) << 8
        self.logger.debug(
            f"ToA-Output before truncation = {bin(counter_val - tdc_num)}")
        self.logger.debug(
            f"ToA-Output after truncation = {(counter_val - tdc_num) & 0x3ff}")
        if plot_data:
            return (time_of_next_counter_edge,
                    copy(self.ctdc._time_bin_edges_with_factors),
                    ctdc_code,
                    residue,
                    amplified_residue,
                    copy(self.ftdc._time_bin_edges_with_factors),
                    ftdc_code,
                    ftdc_num,
                    ctdc_num,
                    counter_val)
        return (counter_val - tdc_num) & 0x3ff
