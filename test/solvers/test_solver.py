from pathlib import Path
from unittest.mock import Mock

import pytest

from fullwave_simulation.conditions import FSAInitialCondition
from fullwave_simulation.constants import FSASimulationParams
from fullwave_simulation.domains import DomainOrganizer
from fullwave_simulation.solvers import Solver
from fullwave_simulation.transducers import ConvexTxWaveTransmitter, SignalReceiver


def build_instance(tmp_path, on_memory):
    work_dir = tmp_path
    simulation_params = Mock(spec=FSASimulationParams)
    domain_organizer = Mock(spce=DomainOrganizer)

    wave_transmitter = Mock(spec=ConvexTxWaveTransmitter)
    signal_receiver = Mock(spec=SignalReceiver)
    initial_condition = Mock(spec=FSAInitialCondition)

    solver = Solver(
        work_dir,
        simulation_params,
        domain_organizer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
        on_memory,
    )
    return solver


# parametrize the pytest with on_memory = True/False
@pytest.mark.parametrize("on_memory", [True, False])
def test_instance(tmp_path, on_memory):
    solver = build_instance(tmp_path, on_memory)
    assert solver.work_dir.exists()
    if on_memory:
        # check if the work_dir is on memory
        assert solver.work_dir.parent == Path("/dev/shm")
