from pathlib import Path

import numpy as np

from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains import Phantom
from fullwave_simulation.utils import test_utils


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
    dX = lambda_ / ppw

    domain = Phantom(
        num_x=num_x,
        num_y=num_y,
        simulation_params=simulation_params,
        material_properties=material_properties,
        dX=dX,
        dY=dX,
    )
    return domain


def test_full_instance():
    domain = build_full_instance()
    assert isinstance(domain, Phantom)


def test_properties():
    test_data_dir = test_utils.get_test_data_dir("domains")
    domain = build_full_instance()
    geometry = domain.geometry
    test_utils.check_variable(
        mat_file_path=test_data_dir / "phantom_geometry.mat",
        var_name="fp_maps",
        test_value=geometry,
    )
