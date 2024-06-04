from pathlib import Path

import numpy as np
from tqdm import tqdm

from fullwave_simulation.conditions import FocusedInitialCondition, FSAInitialCondition
from fullwave_simulation.constants import (
    FocusedSimulationParams,
    FSASimulationParams,
    MaterialProperties,
    MaterialPropertiesFocused,
)
from fullwave_simulation.domains import (
    AbdominalWall,
    Background,
    DomainOrganizer,
    Phantom,
    Scatterer,
    WaterGel,
)
from fullwave_simulation.solvers import FullwaveSolver
from fullwave_simulation.transducers import (
    C52VTransducer,
    ConvexTxWaveTransmitter,
    L125Transducer,
    LinearTxWaveTransmitter,
    SignalReceiver,
)
from fullwave_simulation.utils import test_utils


class FullwaveSolverMod(FullwaveSolver):
    def run(self):
        if self._sequence_type == "fsa":
            self._save_data_for_fsa_beamforming()
        elif self._sequence_type == "focused":
            self._save_data_for_focused_us_beamforming()
        else:
            raise NotImplementedError(f"sequence_type={self._sequence_type} is not implemented")
        result_list = []
        for i_event in tqdm(
            range(self.simulation_params.nevents), desc="running Fullwave simulation"
        ):
            simulation_dir = self.fullwave_input_generator.run(i_event)
            genout = self.fullwave_launcher.run(simulation_dir)
            result_list.append(genout)
            # break added for testing
            break
        return result_list


def setup_inputs_convex_tx_fsa():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)

    background = Background(
        c52v_transducer.num_x,
        c52v_transducer.num_y,
        material_properties,
        simulation_params,
    )
    scatter = Scatterer(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        material_properties=material_properties,
        simulation_params=simulation_params,
        transducer=c52v_transducer,
    )
    csr = 0.035
    background.rho_map = background.rho_map - scatter.rho_map * csr
    water_gel = WaterGel(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        depth=0.0124,
        dY=c52v_transducer.dY,
        material_properties=material_properties,
        simulation_params=simulation_params,
    )
    phantom = Phantom(
        c52v_transducer.num_x,
        c52v_transducer.num_y,
        material_properties,
        simulation_params,
        c52v_transducer.dX,
        c52v_transducer.dY,
    )

    domain_organizer = DomainOrganizer(
        material_properties=material_properties,
    )
    domain_organizer.register_domains(
        [
            background,
            water_gel,
            phantom,
            c52v_transducer.convex_transmitter_map,
        ],
    )
    domain_organizer.construct_domain()

    wave_transmitter = ConvexTxWaveTransmitter(
        c52v_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )
    signal_receiver = SignalReceiver(
        c52v_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
    )

    initial_condition = FSAInitialCondition(
        transducer=c52v_transducer,
        wave_transmitter=wave_transmitter,
    )
    return (
        simulation_params,
        domain_organizer,
        c52v_transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    )


def setup_inputs_linear_tx_focused():
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialPropertiesFocused()

    l125_transducer = L125Transducer(simulation_params, material_properties)

    abdominal_wall = AbdominalWall(
        num_x=l125_transducer.num_x,
        num_y=l125_transducer.num_y,
        crop_depth=0.8e-2,
        # start_depth=0.0124,
        start_depth=0.0,
        # representative_length=0.0124,
        dY=l125_transducer.dY,
        dX=l125_transducer.dX,
        transducer=l125_transducer,
        abdominal_wall_mat_path=Path(
            "fullwave_simulation/domains/data/abdominal_wall/i2365f_etfw1.mat"
        ),
        material_properties=material_properties,
        simulation_params=simulation_params,
        use_smoothing=True,
        skip_i0=False,
        apply_tissue_deformation=False,
        apply_tissue_compression=True,
        background_domain_properties="lung_fluid",
    )

    background = Background(
        abdominal_wall.geometry.shape[0],
        l125_transducer.num_y,
        material_properties,
        simulation_params,
        background_domain_properties="water",
    )

    domain_organizer = DomainOrganizer(
        material_properties=material_properties,
        ignore_non_linearity=True,
        background_domain_properties="lung_fluid",
    )
    domain_organizer.register_domains(
        [
            background,
            abdominal_wall,
        ],
    )
    domain_organizer.construct_domain()

    wave_transmitter = LinearTxWaveTransmitter(
        l125_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )
    signal_receiver = SignalReceiver(
        l125_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
    )

    initial_condition = FocusedInitialCondition(
        transducer=l125_transducer,
        wave_transmitter=wave_transmitter,
        simulation_params=simulation_params,
    )
    return (
        simulation_params,
        domain_organizer,
        l125_transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    )


def build_full_instance_convex_tx_fsa(tmp_path: Path, on_memory: bool) -> FullwaveSolver:
    work_dir = tmp_path
    (
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    ) = setup_inputs_convex_tx_fsa()

    sequence_type = "fsa"
    solver = FullwaveSolverMod(
        work_dir=work_dir,
        simulation_params=simulation_params,
        domain_organizer=domain_organizer,
        transducer=transducer,
        wave_transmitter=wave_transmitter,
        signal_receiver=signal_receiver,
        initial_condition=initial_condition,
        on_memory=on_memory,
        sequence_type=sequence_type,
    )
    return solver


def build_full_instance_linear_tx_focused(tmp_path: Path, on_memory: bool) -> FullwaveSolver:
    work_dir = tmp_path
    (
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    ) = setup_inputs_linear_tx_focused()

    sequence_type = "focused"
    solver = FullwaveSolverMod(
        work_dir=work_dir,
        simulation_params=simulation_params,
        domain_organizer=domain_organizer,
        transducer=transducer,
        wave_transmitter=wave_transmitter,
        signal_receiver=signal_receiver,
        initial_condition=initial_condition,
        on_memory=on_memory,
        sequence_type=sequence_type,
    )
    return solver


def test_run_convex_tx_fsa(tmp_path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    solver = build_full_instance_convex_tx_fsa(tmp_path, on_memory=False)
    solver.run()

    i_event = 0

    simulation_dir = tmp_path / f"{i_event + 1}"

    event_dependent_list = ["icmat"]
    coords_name_list = ["icc", "outc"]
    map_name_list = ["c", "K", "rho", "beta"]
    step_name_list = ["dX", "dY", "dT", "c0"]
    coords_param_list = ["nX", "nY", "nT", "ncoords", "ncoordsout", "nTic", "modT"]
    d_param_list = ["d", "dmap", "ndmap", "dcmap"]

    kappa_name_list = [
        "kappax",
        "kappay",
        "kappau",
        "kappaw",
    ]

    pml_name_list = [
        "apmlu1",
        "bpmlu1",
        "apmlw1",
        "bpmlw1",
        "apmlx1",
        "bpmlx1",
        "apmly1",
        "bpmly1",
        "apmlu2",
        "bpmlu2",
        "apmlw2",
        "bpmlw2",
        "apmlx2",
        "bpmlx2",
        "apmly2",
        "bpmly2",
    ]

    variable_name_list = (
        coords_name_list
        + map_name_list
        + step_name_list
        + coords_param_list
        + d_param_list
        + kappa_name_list
        + pml_name_list
    )

    for var_name in variable_name_list:
        dat_file_path = simulation_dir / f"{var_name}.dat"
        test_dat_file_path = test_data_dir / "test_event_1" / f"{var_name}.dat"
        test_utils.load_and_check_dat_data(
            dat_file_path,
            test_dat_file_path,
            dtype=np.float32,
        )
    for var_name in event_dependent_list:
        dat_file_path = simulation_dir / f"{var_name}.dat"
        test_dat_file_path = test_data_dir / f"test_event_{i_event + 1}" / f"{var_name}.dat"
        test_utils.load_and_check_dat_data(
            dat_file_path,
            test_dat_file_path,
            dtype=np.float32,
        )

    test_simulation_dir = tmp_path / f"{i_event + 1}"
    report_path = test_simulation_dir / "report.txt"
    ground_truth_report_path = test_data_dir / "test_event_1" / "report.txt"
    test_utils.load_and_check_text_data(report_path, ground_truth_report_path)

    genout_path = test_simulation_dir / "genout.dat"
    ground_truth_data_path = test_data_dir / "test_event_1" / "genout.dat"
    test_utils.load_and_check_dat_data(genout_path, ground_truth_data_path)


def test_run_linear_tx_focused_with_abdominal_wall(tmp_path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    solver = build_full_instance_linear_tx_focused(tmp_path, on_memory=False)
    isinstance(solver, FullwaveSolver)
    solver.run()

    i_event = 0

    simulation_dir = tmp_path / f"{i_event + 1}"

    event_dependent_list = ["icmat"]
    coords_name_list = ["icc", "outc"]
    map_name_list = ["c", "K", "rho", "beta"]
    step_name_list = ["dX", "dY", "dT", "c0"]
    # coords_param_list = ["nX", "nY", "nT", "ncoords", "ncoordsout", "nTic", "modT"]
    coords_param_list = ["nX", "nY", "nT", "ncoords", "ncoordsout", "nTic"]
    d_param_list = ["d", "dmap", "ndmap", "dcmap"]

    kappa_name_list = [
        "kappax",
        "kappay",
        "kappau",
        "kappaw",
    ]

    pml_name_list = [
        "apmlu1",
        "bpmlu1",
        "apmlw1",
        "bpmlw1",
        "apmlx1",
        "bpmlx1",
        "apmly1",
        "bpmly1",
        "apmlu2",
        "bpmlu2",
        "apmlw2",
        "bpmlw2",
        "apmlx2",
        "bpmlx2",
        "apmly2",
        "bpmly2",
    ]

    variable_name_list = (
        coords_name_list
        + map_name_list
        + step_name_list
        + coords_param_list
        + d_param_list
        + kappa_name_list
        + pml_name_list
    )

    for var_name in variable_name_list:
        dat_file_path = simulation_dir / f"{var_name}.dat"
        test_dat_file_path = (
            test_data_dir / f"linear_tx_test_event_{i_event +1}" / f"{var_name}.dat"
        )
        test_utils.load_and_check_dat_data(
            dat_file_path,
            test_dat_file_path,
            dtype=np.float32,
        )
    for var_name in event_dependent_list:
        dat_file_path = simulation_dir / f"{var_name}.dat"
        test_dat_file_path = (
            test_data_dir / f"linear_tx_test_event_{i_event + 1}" / f"{var_name}.dat"
        )
        test_utils.load_and_check_dat_data(
            dat_file_path,
            test_dat_file_path,
            dtype=np.float32,
        )

    test_simulation_dir = tmp_path / f"{i_event + 1}"
    # report_path = test_simulation_dir / "report.txt"
    # ground_truth_report_path = test_data_dir / f"linear_tx_test_event_{i_event + 1}" / "report.txt"
    # test_utils.load_and_check_text_data(report_path, ground_truth_report_path)

    genout_path = test_simulation_dir / "genout.dat"
    ground_truth_data_path = test_data_dir / f"linear_tx_test_event_{i_event + 1}" / "genout.dat"
    test_utils.load_and_check_dat_data(genout_path, ground_truth_data_path)
