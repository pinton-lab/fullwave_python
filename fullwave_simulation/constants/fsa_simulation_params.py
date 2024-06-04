import numpy as np

from fullwave_simulation.constants.constant import Constant


class FSASimulationParams(Constant):
    # actual spacing of L12-5 50mm = 0.1953e-3 [m]
    spacing_m = 0.0001953
    # number of active txducer elements (each emit)
    txducer_aperture = 64
    # 64 sequential txrx events
    nevents = 128
    # arbitrarily chosen
    focal_depth = 0.02

    # Basic variables / parameters
    freq_div = 1
    c0 = 1540
    # frequency [MHz]
    f0 = 3.7e6 / freq_div
    # not realistic % radio frequency.
    omega0 = 2 * np.pi * f0
    lambda_ = c0 / f0

    # Simulation grid varables
    # width of simulation field (m). lateral dimension.
    wX = 0.12
    # depth of simulation field (m)
    wY = 0.04 + 0.0124
    # duration of simulation (s).
    # the time how much you want to simulate the propagation (sec)
    dur = wY * 2.3 / c0
    cfl = 0.45

    # pressure in Pa. 100kpa.
    # 2.5Mpa is normal but not in this simulation.
    p0 = 1e5

    num_cycles = 2
    # exponential drop-off of envelope
    drop_off = 1
    ncycles = 2

    # --- aliases ---
    width = wX
    depth = wY
    modT = 7

    # define fnumber
    fnumber = 1.5
    is_fsa = True

    tx_bw = 0.7
