from fullwave_simulation.constants import SimulationParams
from fullwave_simulation.constants.constant import Constant


def build_instance():
    simulation_params = SimulationParams()
    return simulation_params


def test_instance():
    simulation_params = build_instance()
    assert isinstance(simulation_params, SimulationParams)
    assert isinstance(simulation_params, Constant)
