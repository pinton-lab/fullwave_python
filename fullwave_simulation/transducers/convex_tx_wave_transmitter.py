import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.transducers.transducer import Transducer


class ConvexTxWaveTransmitter(Transducer):
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
        if not self.is_fsa:
            Warning("Focused sequence implementation is still experimental for convex transducer")

    def _define_characteristics(self):
        self._determine_grid_params()
        self._define_params_for_focused_sequecne()
        self.transmitter_dd = self._focused_transmit_sequence()
        if self.is_fsa:
            self.transmitter_dd = self._fsa_transmit_sequecne()

    def _determine_grid_params(self):
        # determine other grid vars
        self.nT = np.round(
            self.simulation_params.dur
            * self.simulation_params.f0
            * self.transducer.ppw
            / self.simulation_params.cfl
        )

        self.nT2 = len(
            np.arange(0, self.nT + self.simulation_params.modT, self.simulation_params.modT)
        )

        self.dT = self.transducer.dX / self.material_properties.c0 * self.simulation_params.cfl
        self.fs = (1 / self.dT) / self.simulation_params.modT

    def _define_params_for_focused_sequecne(self):
        self.nTx = self.simulation_params.nevents
        self.bw = self.simulation_params.tx_bw
        self.dep = 0.05 / self.transducer.dY
        self.rad = self.dep + self.transducer.rad
        # define fnumber
        self.fnumber = self.simulation_params.fnumber
        self.fthetanumber = 2 * np.arctan(1 / (2 * self.fnumber))

        # define transmit focal points
        self.bmw = self.transducer.lambda_ * 1.5 / 2 / self.transducer.dY

        self.dTheta = np.arctan2(self.bmw, self.rad)
        self.thetas = self.dTheta * (np.arange((-(self.nTx - 1) / 2), ((self.nTx - 1) / 2) + 1))

    def _focused_transmit_sequence(self):
        self.foc_origs = (
            np.array([np.transpose(np.sin(self.thetas)), np.transpose(np.cos(self.thetas))])
            * self.transducer.rad
        ).T + self.transducer.cen
        self.focs = (
            np.array([np.transpose(np.sin(self.thetas)), np.transpose(np.cos(self.thetas))])
            * self.rad
        ).T + self.transducer.cen
        # generate focal delays
        transmitter_dd = self._focus_delays(
            self.focs, self.transducer.incoords, self.simulation_params.cfl, 0
        )
        transmitter_dd = transmitter_dd - transmitter_dd.min()
        # each transducer element has a self delay, that delay is applied to all subelements
        for tt in range(self.transducer.npx):
            idt = np.where(self.transducer.incoords[:, 3] == tt + 1)[0]
            transmitter_dd[idt, :] = np.ones((len(idt), 1)) * np.mean(transmitter_dd[idt, :], 0)
        return transmitter_dd

    def _fsa_transmit_sequecne(self):
        transmitter_dd = np.zeros(
            (
                self.transducer.incoords.shape[0],
                self.simulation_params.nevents,
            )
        )
        return transmitter_dd

    def _focus_delays(self, focs, coords, cfl, varargin):
        dds = np.zeros((coords.shape[0], focs.shape[0]))
        for n in range(focs.shape[0]):
            idy = focs[n, 0]
            idz = focs[n, 1]
            dd = np.sqrt((coords[:, 0] - idy) ** 2 + (coords[:, 1] - idz) ** 2)
            dd = -np.round(dd / cfl)
            mdd = np.amin(dd)
            optargin = np.array(varargin).ndim + 1
            if optargin == 1:
                mdd = varargin
            dd = dd - mdd
            dds[:, n] = dd
        return dds

    def generate_transmit_pulse(self):
        nTic = np.round(
            np.amax(np.amax(self.transmitter_dd))
            + self.simulation_params.ncycles * 5 * self.transducer.ppw / self.simulation_params.cfl
        )
        t = (
            (np.arange(1, nTic + 1)) / self.nT * self.simulation_params.dur
            - self.simulation_params.ncycles / self.simulation_params.omega0 * 2 * np.pi
        )
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
        return nTic, icvec
