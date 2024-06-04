import numpy as np

from fullwave_simulation.conditions import Condition
from fullwave_simulation.transducers.transducer import Transducer
from fullwave_simulation.utils import utils


class FocusedInitialCondition(Condition):
    def __init__(
        self,
        transducer: Transducer,
        wave_transmitter,
        simulation_params,
    ):
        super().__init__()
        self.transducer = transducer
        self.wave_transmitter = wave_transmitter
        self.foc = self.wave_transmitter.foc
        self.fcen = self.wave_transmitter.fcen
        self.fnumber = self.wave_transmitter.fnumber
        self.incoords = self.transducer.incoords
        self.simulation_params = simulation_params
        self.txducer_aperture = self.simulation_params.txducer_aperture
        self.beam_spacing = self.transducer.beam_spacing

        self._c0 = self.simulation_params.c0
        self._omega0 = self.simulation_params.omega0
        # self._nX = self.transducer.num_x
        # self._nY = self.transducer.num_y
        self._duration = self.simulation_params.dur
        self._ppw = self.transducer.ppw
        self._cfl = self.simulation_params.cfl
        # self._dT = self.wave_transmitter.dT

        self._set_field_params()

    def _set_field_params(self):
        lambda_ = self._c0 / self._omega0 * 2 * np.pi
        # self._nXe = self._nX + 2 * (self._num_body + self._m)
        # self._nYe = self._nY + 2 * (self._num_body + self._m)
        self._nT = int(np.round(self._duration * self._c0 / lambda_ * self._ppw / self._cfl))
        #  Number of time samples in output after downsampling
        self.nT2 = len(np.arange(0, self._nT, self.simulation_params.modT))
        # self._nTic = self.initial_condition.nTic

    def generate_icmat(self):
        icmat = np.zeros((self.transducer.incoords.shape[0], self._nT))
        icvec = self.wave_transmitter.generate_transmit_pulse(i=0)

        icmat[0 : int(self.incoords.shape[0] / 3), :] = self._focus_coords(
            self.fcen[0],
            self.fcen[1],
            self.incoords[0 : int(self.incoords.shape[0] / 3), :],
            icvec,
            self.simulation_params.cfl,
        )
        icvec = self.wave_transmitter.generate_transmit_pulse(i=1)
        icmat[
            int(self.incoords.shape[0] / 3) : int(self.incoords.shape[0] / 3 * 2), :
        ] = self._focus_coords(
            self.fcen[0],
            self.fcen[1],
            self.incoords[
                int(self.incoords.shape[0] / 3) : int(self.incoords.shape[0] / 3 * 2), :
            ],
            icvec,
            self.simulation_params.cfl,
        )

        icvec = self.wave_transmitter.generate_transmit_pulse(i=2)
        icmat[
            int(self.incoords.shape[0] / 3 * 2) : int(self.incoords.shape[0] + 1), :
        ] = self._focus_coords(
            self.fcen[0],
            self.fcen[1],
            self.incoords[
                int(self.incoords.shape[0] / 3 * 2) : int(self.incoords.shape[0] + 1), :
            ],
            icvec,
            self.simulation_params.cfl,
        )

        # 1st
        icmat[
            0 : utils.matlab_round(self.transducer.num_x / 2)
            - utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1,
            :,
        ] = 0
        icmat[
            utils.matlab_round(self.transducer.num_x / 2)
            + utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1 : self.transducer.num_x,
            :,
        ] = 0

        # 2nd
        icmat[
            self.transducer.num_x
            + 1
            - 1 : self.transducer.num_x
            + utils.matlab_round(self.transducer.num_x / 2)
            - utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1,
            :,
        ] = 0
        icmat[
            self.transducer.num_x
            + utils.matlab_round(self.transducer.num_x / 2)
            + utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1 : 2 * self.transducer.num_x,
            :,
        ] = 0

        # 3rd
        icmat[
            2 * self.transducer.num_x
            + 1
            - 1 : 2 * self.transducer.num_x
            + utils.matlab_round(self.transducer.num_x / 2)
            - utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1,
            :,
        ] = 0
        icmat[
            2 * self.transducer.num_x
            + utils.matlab_round(self.transducer.num_x / 2)
            + utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1 : 3 * self.transducer.num_x,
            :,
        ] = 0
        self.nTic = icmat.shape[1]

        return icmat

    def _focus_coords(self, idy, idz, coords, icvec, cfl):
        #############################################################
        # GIANMARCO PINTON
        # Originally written: NOV 13, 2013 by Gianmarco Pinton
        # pythonized: Oct 7, 2023 by Masashi Sode
        # Focus coordinates
        # mdd: offset in time pixels
        #############################################################
        dd = np.sqrt((coords[:, 0] - idy) ** 2 + (coords[:, 1] - idz) ** 2)
        dd = -(dd / cfl)
        dd = utils.matlab_round(dd - np.min(dd))  # changed Oct15-2020 from dd = dd-min(dd)

        icmat = np.zeros((coords.shape[0], len(icvec)))
        for i in range(coords.shape[0]):
            icmat[i, int(dd[i]) :] = icvec[0 : -int(dd[i]) or None]
        return icmat
