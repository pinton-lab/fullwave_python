import numpy as np

from fullwave_simulation.conditions import Condition
from fullwave_simulation.transducers.transducer import Transducer
from fullwave_simulation.utils import utils


class PlaneWaveInitialCondition(Condition):
    def __init__(
        self,
        # is_fsa: bool,
        transducer: Transducer,
        wave_transmitter: Transducer,
        simulation_params,
    ):
        super().__init__()
        # self.is_fsa = is_fsa
        self.transducer = transducer
        self.wave_transmitter = wave_transmitter
        # self.nTic, self.icvec = self.wave_transmitter.generate_transmit_pulse()
        self.simulation_params = simulation_params
        self._n_angles = self.simulation_params.n_angles
        self._d_theta = self.simulation_params.d_theta
        self._c0 = self.simulation_params.c0

        # self.foc = self.wave_transmitter.foc
        # self.fcen = self.wave_transmitter.fcen
        # self.fnumber = self.wave_transmitter.fnumber
        self.incoords = self.transducer.incoords
        # self.txducer_aperture = self.simulation_params.txducer_aperture
        # self.beam_spacing = self.transducer.beam_spacing

        self._c0 = self.simulation_params.c0
        self._omega0 = self.simulation_params.omega0
        # self._nX = self.transducer.num_x
        # self._nY = self.transducer.num_y
        self._duration = self.simulation_params.dur
        self._ppw = self.transducer.ppw
        self._cfl = self.simulation_params.cfl
        # self._dT = self.wave_transmitter.dT
        self._set_field_params()

        t = self.wave_transmitter.generate_t(i=0)
        self.nTic = t.shape[0]

    def _set_field_params(self):
        lambda_ = self._c0 / self._omega0 * 2 * np.pi
        # self._nXe = self._nX + 2 * (self._num_body + self._m)
        # self._nYe = self._nY + 2 * (self._num_body + self._m)
        self._nT = int(np.round(self._duration * self._c0 / lambda_ * self._ppw / self._cfl))
        #  Number of time samples in output after downsampling
        self.nT2 = len(np.arange(0, self._nT, self.simulation_params.modT))
        # self._nTic = self.initial_condition.nTic

    def generate_icmat(self, i_angle: int):
        # theta=(n-(nangles+1)/2)*dtheta

        # fcen=[round(1e6/dY)*sin(theta) round(1e6/dY)]; % center of focus
        # t = (0:nT-1)/nT*duration-ncycles/omega0*2*pi;
        # icvec = exp(-(1.05*t*omega0/(ncycles*pi)).^(2*dur)).*sin(t*omega0)*p0;
        # plot(icvec)
        # icmat=repmat(icvec,size(incoords,1)/8,1);
        # for k=2:8
        #   t=t-dX/c0; icvec = exp(-(1.05*t*omega0/(ncycles*pi)).^(2*dur)).*sin(t*omega0)*p0;
        #   icmat=[icmat' repmat(icvec,size(incoords,1)/8,1)']';
        # end
        # icmat = focusCoords(fcen(1),fcen(2),incoords,icvec,cfl);
        theta = ((i_angle + 1) - (self._n_angles + 1) / 2) * self._d_theta

        focus_center = np.array(
            (
                utils.matlab_round(1e6 / self.transducer.dX) * np.sin(theta),
                utils.matlab_round(1e6 / self.transducer.dX),
            )
        )

        t = self.wave_transmitter.generate_t(i=0)
        icvec = self.wave_transmitter.generate_transmit_pulse_from_t(t)

        assert self.incoords.shape[0] % 8 == 0
        icmat = np.tile(icvec, (self.incoords.shape[0] // 8, 1))

        for i in range(1, 8):
            t = t - self.transducer.dX / self._c0
            icvec = self.wave_transmitter.generate_transmit_pulse_from_t(t)
            icmat = np.concatenate([icmat, np.tile(icvec, (self.incoords.shape[0] // 8, 1))], 0)

        icmat = self._focus_coords(
            focus_center[0],
            focus_center[1],
            self.incoords,
            icvec,
            self._cfl,
        )
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
