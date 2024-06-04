import tempfile
from pathlib import Path

import fullwave_simulation
from fullwave_simulation.conditions import FSAInitialCondition
from fullwave_simulation.constants import FSASimulationParams
from fullwave_simulation.domains import DomainOrganizer
from fullwave_simulation.transducers import (
    C52VTransducer,
    ConvexTxWaveTransmitter,
    SignalReceiver,
)


class Solver:
    def __init__(
        self,
        work_dir: Path,
        simulation_params: FSASimulationParams,
        #
        domain_organizer: DomainOrganizer,
        transducer: C52VTransducer,
        wave_transmitter: ConvexTxWaveTransmitter,
        signal_receiver: SignalReceiver,
        #
        initial_condition: FSAInitialCondition,
        on_memory=True,
        home_dir=Path(fullwave_simulation.__file__).parent.parent,
        path_fullwave_simulation_bin=(
            Path(fullwave_simulation.__file__).parent.parent
            / "fullwave_simulation/bin/fullwave2_try6_nln_relaxing_pzero_rebuild3"
        ),
    ):
        assert home_dir.exists()
        assert path_fullwave_simulation_bin.exists()
        self.home_dir = home_dir
        self.work_dir_org = work_dir
        self.path_fullwave_simulation_bin = path_fullwave_simulation_bin
        self.simulation_params = simulation_params
        self.domain_organizer = domain_organizer
        self.transducer = transducer
        self.wave_transmitter = wave_transmitter
        self.signal_receiver = signal_receiver
        self.initial_condition = initial_condition
        self.on_memory = on_memory
        if self.on_memory:
            # make an additional workdir on memory under /dev/shm to speed up the I/O.
            self.temp_dir_object = tempfile.TemporaryDirectory(dir="/dev/shm")
            self.work_dir = Path(self.temp_dir_object.name)
            print(f"work_dir is on memory: {self.work_dir}")
        else:
            self.work_dir = work_dir
