from fullwave_simulation.constants import FSASimulationParams
from fullwave_simulation.constants.constant import Constant


def build_instance():
    simulation_params = FSASimulationParams()
    return simulation_params


def test_instance():
    simulation_params = build_instance()
    assert isinstance(simulation_params, FSASimulationParams)
    assert isinstance(simulation_params, Constant)
