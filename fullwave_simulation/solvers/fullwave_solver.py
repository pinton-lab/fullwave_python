import shutil
from pathlib import Path
from typing import Union

import numpy as np
import scipy
from tqdm import tqdm

import fullwave_simulation
from fullwave_simulation.conditions import FSAInitialCondition
from fullwave_simulation.constants import FSASimulationParams
from fullwave_simulation.domains import DomainOrganizer
from fullwave_simulation.solvers import Solver
from fullwave_simulation.solvers.focused_input_generator import FocusedInputGenerator
from fullwave_simulation.solvers.fsa_input_generator import FSAInputGenerator
from fullwave_simulation.solvers.fullwave_launcher import FullwaveLauncher
from fullwave_simulation.solvers.plane_wave_input_generator import (
    PlaneWaveInputGenerator,
)
from fullwave_simulation.transducers import (
    C52VTransducer,
    ConvexTxWaveTransmitter,
    SignalReceiver,
)


class FullwaveSolver(Solver):
    def __init__(
        self,
        work_dir: Path,
        #
        simulation_params: FSASimulationParams,
        #
        domain_organizer: DomainOrganizer,
        transducer: C52VTransducer,
        wave_transmitter: ConvexTxWaveTransmitter,
        signal_receiver: SignalReceiver,
        #
        initial_condition,
        sequence_type: str = "fsa",
        on_memory=False,
        home_dir=Path(fullwave_simulation.__file__).parent.parent,
        path_fullwave_simulation_bin=(
            Path(fullwave_simulation.__file__).parent.parent
            / "fullwave_simulation/bin/fullwave2_try6_nln_relaxing_pzero_rebuild3"
        ),
    ):
        self._sequence_type = sequence_type
        super().__init__(
            work_dir=work_dir,
            path_fullwave_simulation_bin=path_fullwave_simulation_bin,
            simulation_params=simulation_params,
            domain_organizer=domain_organizer,
            transducer=transducer,
            wave_transmitter=wave_transmitter,
            signal_receiver=signal_receiver,
            initial_condition=initial_condition,
            on_memory=on_memory,
            home_dir=home_dir,
        )

        self.fullwave_launcher = FullwaveLauncher(
            self.home_dir,
            self.path_fullwave_simulation_bin,
        )
        self.fullwave_input_generator: Union[FSAInputGenerator, FocusedInputGenerator]

        if self._sequence_type == "fsa":
            self.fullwave_input_generator = FSAInputGenerator(
                work_dir=self.work_dir,
                simulation_params=simulation_params,
                domain_organizer=domain_organizer,
                transducer=transducer,
                wave_transmitter=wave_transmitter,
                signal_receiver=signal_receiver,
                initial_condition=initial_condition,
            )
        elif self._sequence_type == "focused":
            self.fullwave_input_generator = FocusedInputGenerator(
                work_dir=self.work_dir,
                simulation_params=simulation_params,
                domain_organizer=domain_organizer,
                transducer=transducer,
                wave_transmitter=wave_transmitter,
                signal_receiver=signal_receiver,
                initial_condition=initial_condition,
            )
        elif self._sequence_type == "plane":
            self.fullwave_input_generator = PlaneWaveInputGenerator(
                work_dir=self.work_dir,
                simulation_params=simulation_params,
                domain_organizer=domain_organizer,
                transducer=transducer,
                wave_transmitter=wave_transmitter,
                signal_receiver=signal_receiver,
                initial_condition=initial_condition,
            )
        else:
            raise NotImplementedError(f"sequence_type={self._sequence_type} is not implemented")

    def run(self):
        if self._sequence_type == "fsa":
            self._save_data_for_fsa_beamforming()
        elif self._sequence_type == "focused":
            self._save_data_for_focused_us_beamforming()
        elif self._sequence_type == "plane":
            self._save_data_for_plane_wave_beamforming()
        else:
            raise NotImplementedError(f"sequence_type={self._sequence_type} is not implemented")
        result_list = []
        for i_event in tqdm(
            range(self.simulation_params.nevents), desc="running Fullwave simulation"
        ):
            simulation_dir = self.fullwave_input_generator.run(i_event)
            genout = self.fullwave_launcher.run(simulation_dir)
            result_list.append(genout)
            if self.on_memory:
                genout_path = simulation_dir / "genout.dat"
                output_dir = self.work_dir_org / f"{i_event}"
                output_dir.mkdir(exist_ok=True, parents=True)
                shutil.copy(src=genout_path, dst=output_dir / "genout.dat")
                # self.temp_dir_object.cleanup()
        return result_list

    def _save_data_for_fsa_beamforming(self):
        # nT2, xdc.npx, tx.nTx, xdc.nOutPx, xdc.outcoords,
        # fs, f0, c0, tx.bw, xdc.rad, xdc.cen, dY,
        # lambda, nY, dT, modT, xdc.outcoords2, xdc.incoords2
        nT2 = self.fullwave_input_generator.nT2
        npx = self.transducer.npx
        nTx = self.simulation_params.nevents
        nOutPx = self.transducer.nOutPx
        outcoords = self.transducer.outcoords
        fs = self.wave_transmitter.fs
        f0 = self.simulation_params.f0
        c0 = self.simulation_params.c0
        bw = self.wave_transmitter.bw
        rad = self.transducer.rad
        cen = self.transducer.cen
        dX = self.transducer.dX
        dY = self.transducer.dY
        lambda_ = self.transducer.lambda_
        nX = self.transducer.num_x
        nY = self.transducer.num_y
        dT = self.wave_transmitter.dT
        modT = self.simulation_params.modT
        outcoords2 = self.transducer.outcoords2
        incoords2 = self.transducer.incoords2
        fnumber = self.simulation_params.fnumber
        cmap = self.domain_organizer.constructed_domain_dict["c_map"]

        export_dict = {
            "nT2": nT2,
            "npx": npx,
            "nTx": nTx,
            "nOutPx": nOutPx,
            "outcoords": outcoords.astype(np.float32),
            "fs": np.array(fs).astype(np.float32),
            "f0": np.array(f0).astype(np.float32),
            "c0": np.array(c0).astype(np.float32),
            "bw": np.array(bw).astype(np.float32),
            "rad": np.array(rad).astype(np.float32),
            "cen": np.array(cen).astype(np.float32),
            "dX": np.array(dX).astype(np.float32),
            "dY": np.array(dY).astype(np.float32),
            "lambda": np.array(lambda_).astype(np.float32),
            "nX": np.array(nX).astype(np.int32),
            "nY": np.array(nY).astype(np.int32),
            "dT": np.array(dT).astype(np.float32),
            "modT": np.array(modT).astype(np.int32),
            "outcoords2": outcoords2.astype(np.float32),
            "incoords2": incoords2.astype(np.float32),
            "fnumber": np.array(fnumber).astype(np.float32),
            "cmap": cmap.astype(np.float32),
        }
        scipy.io.savemat(self.work_dir / "launcher_workspace.mat", export_dict)

    def _save_data_for_focused_us_beamforming(self):
        # nT2, xdc.npx, tx.nTx, xdc.nOutPx, xdc.outcoords,
        # fs, f0, c0, tx.bw, xdc.rad, xdc.cen, dY,
        # lambda, nY, dT, modT, xdc.outcoords2, xdc.incoords2
        nT = self.wave_transmitter.nT
        nT2 = self.fullwave_input_generator.nT2
        dT = self.wave_transmitter.dT
        # npx = self.transducer.npx
        nevents = self.simulation_params.nevents
        # nOutPx = self.transducer.nOutPx
        outcoords = self.transducer.outcoords
        # fs = self.wave_transmitter.fs
        f0 = self.simulation_params.f0
        c0 = self.simulation_params.c0
        # bw = self.wave_transmitter.bw
        # rad = self.transducer.rad
        # cen = self.transducer.cen
        dX = self.transducer.dX
        dY = self.transducer.dY
        lambda_ = self.transducer.lambda_
        nX = self.transducer.num_x
        nY = self.transducer.num_y
        # dT = self.wave_transmitter.dT
        modT = self.simulation_params.modT
        # outcoords2 = self.transducer.outcoords2
        # incoords2 = self.transducer.incoords2
        fnumber = self.wave_transmitter.fnumber
        cmap = self.domain_organizer.constructed_domain_dict["c_map"]
        spacing = self.transducer.beam_spacing

        export_dict = {
            "nT": nT,
            "nT2": nT2,
            # "npx": npx,
            "nevents": nevents,
            "spacing": spacing,
            # "nOutPx": nOutPx,
            "outcoords": outcoords.astype(np.float32),
            # "fs": np.array(fs).astype(np.float32),
            "f0": np.array(f0).astype(np.float32),
            "c0": np.array(c0).astype(np.float32),
            # "bw": np.array(bw).astype(np.float32),
            # "rad": np.array(rad).astype(np.float32),
            # "cen": np.array(cen).astype(np.float32),
            "dX": np.array(dX).astype(np.float32),
            "dY": np.array(dY).astype(np.float32),
            "lambda": np.array(lambda_).astype(np.float32),
            "nX": np.array(nX).astype(np.int32),
            "nY": np.array(nY).astype(np.int32),
            # "dT": np.array(dT).astype(np.float32),
            "modT": np.array(modT).astype(np.int32),
            # "outcoords2": outcoords2.astype(np.float32),
            # "incoords2": incoords2.astype(np.float32),
            "fnumber": np.array(fnumber).astype(np.float32),
            "cmap": cmap.astype(np.float32),
            "dT": np.array(dT).astype(np.float32),
        }
        scipy.io.savemat(self.work_dir / "launcher_workspace.mat", export_dict)

    def _save_data_for_plane_wave_beamforming(self):
        # nT2, xdc.npx, tx.nTx, xdc.nOutPx, xdc.outcoords,
        # fs, f0, c0, tx.bw, xdc.rad, xdc.cen, dY,
        # lambda, nY, dT, modT, xdc.outcoords2, xdc.incoords2
        nT = self.wave_transmitter.nT
        nT2 = self.fullwave_input_generator.nT2
        dT = self.wave_transmitter.dT
        # npx = self.transducer.npx
        nevents = self.simulation_params.nevents
        # nOutPx = self.transducer.nOutPx
        outcoords = self.transducer.outcoords
        # fs = self.wave_transmitter.fs
        f0 = self.simulation_params.f0
        c0 = self.simulation_params.c0
        # bw = self.wave_transmitter.bw
        # rad = self.transducer.rad
        # cen = self.transducer.cen
        dX = self.transducer.dX
        dY = self.transducer.dY
        lambda_ = self.transducer.lambda_
        nX = self.transducer.num_x
        nY = self.transducer.num_y
        # dT = self.wave_transmitter.dT
        modT = self.simulation_params.modT
        # outcoords2 = self.transducer.outcoords2
        # incoords2 = self.transducer.incoords2
        fnumber = self.wave_transmitter.fnumber
        cmap = self.domain_organizer.constructed_domain_dict["c_map"]
        spacing = self.transducer.beam_spacing

        export_dict = {
            "nT": nT,
            "nT2": nT2,
            # "npx": npx,
            "nevents": nevents,
            "spacing": spacing,
            # "nOutPx": nOutPx,
            "outcoords": outcoords.astype(np.float32),
            # "fs": np.array(fs).astype(np.float32),
            "f0": np.array(f0).astype(np.float32),
            "c0": np.array(c0).astype(np.float32),
            # "bw": np.array(bw).astype(np.float32),
            # "rad": np.array(rad).astype(np.float32),
            # "cen": np.array(cen).astype(np.float32),
            "dX": np.array(dX).astype(np.float32),
            "dY": np.array(dY).astype(np.float32),
            "lambda": np.array(lambda_).astype(np.float32),
            "nX": np.array(nX).astype(np.int32),
            "nY": np.array(nY).astype(np.int32),
            # "dT": np.array(dT).astype(np.float32),
            "modT": np.array(modT).astype(np.int32),
            # "outcoords2": outcoords2.astype(np.float32),
            # "incoords2": incoords2.astype(np.float32),
            "fnumber": np.array(fnumber).astype(np.float32),
            "cmap": cmap.astype(np.float32),
            "dT": np.array(dT).astype(np.float32),
        }
        scipy.io.savemat(self.work_dir / "launcher_workspace.mat", export_dict)
