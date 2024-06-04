from pathlib import Path

import numpy as np

from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains import WaterGel


def build_full_instance():
    num_x = 100
    num_y = 100
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    freq_div = 1
    c0 = 1540
    f0 = 3700000.0 / freq_div
    lambda_ = c0 / f0
    wX = 0.12
    wY = 0.04 + 0.0124

    ptch = 0.000508
    subelem_ptch = 15 / freq_div

    ppw = lambda_ / (ptch / subelem_ptch)

    num_x = np.round(wX / lambda_ * ppw).astype(int)
    num_y = np.round(wY / lambda_ * ppw).astype(int)
    dY = lambda_ / ppw
    depth = 12.4e-3

    domain = WaterGel(
        num_x=num_x,
        num_y=num_y,
        depth=depth,
        dY=dY,
        simulation_params=simulation_params,
        material_properties=material_properties,
    )
    return domain


def test_full_instance():
    domain = build_full_instance()
    assert isinstance(domain, WaterGel)


def test_properties():
    domain = build_full_instance()
    geometry = domain.geometry
    rho_map = domain.rho_map
    c_map = domain.c_map
    a_map = domain.a_map
    beta_map = domain.beta_map
    assert rho_map.max() == domain.material_properties.water["rho0"]
    assert c_map.max() == domain.material_properties.water["c0"]
    assert a_map.max() == domain.material_properties.water["alpha"]
    assert beta_map.max() == domain.material_properties.beta0
