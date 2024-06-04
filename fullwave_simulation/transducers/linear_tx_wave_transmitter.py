import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.transducers.transducer import Transducer
from fullwave_simulation.utils import utils


class LinearTxWaveTransmitter(Transducer):
    def __init__(
        self,
        transducer: Transducer,
        simulation_params: Constant,
        material_properties: Constant,
        is_fsa=False,
    ):
        super().__init__(
            simulation_params=simulation_params,
            material_properties=material_properties,
        )
        self.transducer = transducer
        self.is_fsa = is_fsa
        self._define_characteristics()
        if self.is_fsa:
            raise NotImplementedError("FSA not implemented for linear transducer")

    def _define_characteristics(self):
        self._determine_grid_params()
        self._define_params_for_focused_sequecne()

    def _determine_grid_params(self):
        self._c0 = self.simulation_params.c0
        self._omega0 = self.simulation_params.omega0
        # self._nX = self.transducer.num_x
        # self._nY = self.transducer.num_y
        self._duration = self.simulation_params.dur
        self._ppw = self.transducer.ppw
        self._cfl = self.simulation_params.cfl
        # self._dT = self.wave_transmitter.dT

        lambda_ = self._c0 / self._omega0 * 2 * np.pi
        self.dT = self.transducer.dX / self.material_properties.c0 * self.simulation_params.cfl
        # self._nXe = self._nX + 2 * (self._num_body + self._m)
        # self._nYe = self._nY + 2 * (self._num_body + self._m)
        self.nT = int(np.round(self._duration * self._c0 / lambda_ * self._ppw / self._cfl))
        #  Number of time samples in output after downsampling
        self.nT2 = len(np.arange(0, self.nT, self.simulation_params.modT))
        # self._nTic = self.initial_condition.nTic

    def _define_params_for_focused_sequecne(self):
        self.foc = utils.matlab_round(self.simulation_params.focal_depth / self.transducer.dX)
        self.fcen = np.array((utils.matlab_round(self.transducer.num_x / 2), self.foc))
        self.beam_spacing = utils.matlab_round(
            self.simulation_params.spacing_m / self.transducer.dX
        )
        self.fnumber = self.simulation_params.focal_depth / (
            self.simulation_params.txducer_aperture * self.beam_spacing * self.transducer.dX
        )

    def generate_t(self, i):
        t = (
            (np.arange(0, self.nT)) / self.nT * self.simulation_params.dur
            - self.simulation_params.ncycles / self.simulation_params.omega0 * 2 * np.pi
        )
        t = t - (self.dT / self.simulation_params.cfl) * i
        return t

    def generate_transmit_pulse(self, i):
        t = self.generate_t(i)
        icvec = (
            np.multiply(
                np.exp(
                    -(
                        (
                            1.05
                            * t
                            * self.simulation_params.omega0
                            / (self.simulation_params.ncycles * np.pi)
                        )
                        ** (2 * self.simulation_params.drop_off)
                    )
                ),
                np.sin(t * self.simulation_params.omega0),
            )
            * self.simulation_params.p0
        )
        return icvec

    def generate_transmit_pulse_from_t(self, t):
        icvec = (
            np.multiply(
                np.exp(
                    -(
                        (
                            1.05
                            * t
                            * self.simulation_params.omega0
                            / (self.simulation_params.ncycles * np.pi)
                        )
                        ** (2 * self.simulation_params.drop_off)
                    )
                ),
                np.sin(t * self.simulation_params.omega0),
            )
            * self.simulation_params.p0
        )
        return icvec
