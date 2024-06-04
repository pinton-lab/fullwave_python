import numpy as np

from fullwave_simulation.constants.constant import Constant


class FocusedSimulationParams(Constant):
    # --- for focused sequence ---
    # arbitrarily chosen
    focal_depth = 2e-2
    # actual spacing of L12-5 50mm = 0.1953e-4 [m]
    spacing_m = 1.953e-4
    # number of active txducer elements (each emit)
    txducer_aperture = 64
    # walking aperture, sequential txrx events
    nelements = 192
    nevents = nelements - txducer_aperture

    # --- Basic variables / parameters ---
    # speed of sound (m/s)
    c0 = 1540
    # frequency [MHz]
    f0 = 6.25e6
    # pressure in Pa.
    p0 = 2.5e6

    # number of points per spatial wavelength
    ppw = 12
    # Courant-Friedrichs-Levi condition
    cfl = 0.5

    # width of simulation field (m). lateral dimension.
    wX = 2.5e-2
    # depth of simulation field (m)
    wY = 3.5e-2
    # wY = 4.0e-2

    # duration of simulation (s).
    # the time how much you want to simulate the propagation (sec)
    # dur = wY * 2.3 / c0
    dur = 15.6e-2 / c0

    # --- initial conditions ---
    # number of cycles in pulse
    ncycles = 2
    num_cycles = ncycles
    # exponential drop-off of envelope
    drop_off = 2

    # --- aliases ---
    width = wX
    depth = wY
    modT = 7

    # define fnumber
    fnumber = 1.5
    is_fsa = False

    tx_bw = 0.7

    @property
    def omega0(self):
        return 2 * np.pi * self.f0

    @property
    def lambda_(self):
        return self.c0 / self.f0
