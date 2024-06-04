import numpy as np

from fullwave_simulation.conditions.condition import Condition
from fullwave_simulation.transducers.transducer import Transducer


class FSAInitialCondition(Condition):
    def __init__(
        self,
        # is_fsa: bool,
        transducer: Transducer,
        wave_transmitter: Transducer,
    ):
        super().__init__()
        # self.is_fsa = is_fsa
        self.transducer = transducer
        self.wave_transmitter = wave_transmitter
        self.nTic, self.icvec = self.wave_transmitter.generate_transmit_pulse()

    def generate_icmat(self, event_id: int):
        icmat = self._focus_coords_dd(
            np.round(self.wave_transmitter.transmitter_dd[:, event_id]), self.icvec, 0
        )
        off_idx = self.transducer.incoords[:, 3] != event_id + 1
        icmat[off_idx, :] = 0
        return icmat

    def _focus_coords_dd(self, delay, icvec, varargin):
        #############################################################
        # GIANMARCO PINTON
        # WRITTEN: NOV 13, 2013
        # LAST MODIFIED: JAN 4, 2021
        # Focus coordinates
        # mdd: offset in time pixels
        #############################################################
        mdd = np.amin(np.amin(delay))
        optargin = np.array(varargin).ndim + 1
        if optargin == 1:
            if varargin < mdd:
                mdd = varargin

        delay = delay - mdd
        icmat = np.zeros((len(delay), len(icvec)))
        for i, delay_i in enumerate(delay):
            icmat[i, int(delay_i) :] = icvec[0 : -int(delay_i) or None]
        return icmat
