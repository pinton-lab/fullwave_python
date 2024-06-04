import os
import subprocess
from pathlib import Path

import fullwave_simulation
from fullwave_simulation.utils import test_utils


class FullwaveLauncher:
    def __init__(
        self,
        home_dir=Path(fullwave_simulation.__file__).parent.parent,
        path_fullwave_simulation_bin=(
            Path(fullwave_simulation.__file__).parent.parent
            / "fullwave_simulation/bin/fullwave2_try6_nln_relaxing_pzero_rebuild3"
        ),
    ):
        self._home_dir = home_dir
        self._path_fullwave_simulation_bin = path_fullwave_simulation_bin
        assert self._home_dir.exists()
        assert self._path_fullwave_simulation_bin.exists()

    def run(self, simulation_dir):
        os.chdir(simulation_dir.absolute())
        command = [
            f"{(simulation_dir.absolute() / self._path_fullwave_simulation_bin.name)}",
        ]
        with open(simulation_dir.absolute() / "report.txt", "w", encoding="utf-8") as file:
            subprocess.call(command, stdout=file)
        os.chdir(self._home_dir)
        genout = test_utils.load_dat_data(simulation_dir.absolute() / "genout.dat")
        return genout
