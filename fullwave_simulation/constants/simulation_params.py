import numpy as np

from fullwave_simulation.constants.constant import Constant


class SimulationParams(Constant):
    # actual spacing of L12-5 50mm = 0.1953e-3 [m]
    spacing_m = 0.0001953
    # number of active txducer elements (each emit)
    txducer_aperture = 64
    # 64 sequential txrx events
    nevents = 64
    # arbitrarily chosen
    focal_depth = 0.02

    c0 = 1540
    # frequency [MHz]
    f0 = 3e6
    # not realistic % radio frequency.
    omega0 = 2 * np.pi * f0

    # width of simulation field (m). lateral dimension.
    wX = txducer_aperture * spacing_m + 2 * 0.013
    # depth of simulation field (m)
    wY = 0.035
    # duration of simulation (s).
    # the time how much you want to simulate the propagation (sec)
    duration = 3.5 * wY / c0

    # pressure in Pa. 100kpa.
    # 2.5Mpa is normal but not in this simulation.
    p0 = 1e5

    # number of cycles in tx pulse
    num_cycles = 2
    # exponential drop-off of envelope
    drop_off = 2
    num_scat = 10

    # number of points per spatial wavelength.
    ppw = 12
    # Courant-Friedrichs-Levi condition.
    # it makes the simulation faster than actual process.
    cfl = 0.5

    # --- aliases ---
    width = wX
    depth = wY

    rho0 = 1000
    a0 = 0.5
    beta0 = 0
