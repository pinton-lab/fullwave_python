from fullwave_simulation.constants import FocusedSimulationParams
from fullwave_simulation.constants.constant import Constant


def build_instance():
    simulation_params = FocusedSimulationParams()
    return simulation_params


def test_instance():
    simulation_params = build_instance()
    assert isinstance(simulation_params, FocusedSimulationParams)
    assert isinstance(simulation_params, Constant)
