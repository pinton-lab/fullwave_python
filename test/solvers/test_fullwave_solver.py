from pathlib import Path

import numpy as np
import pytest
import scipy

from fullwave_simulation.conditions import FSAInitialCondition
from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains import (
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
    SignalReceiver,
)
from fullwave_simulation.utils import test_utils


def setup_inputs():
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

    domain_organizer = DomainOrganizer(material_properties=material_properties)
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


def build_full_instance(tmp_path: Path, on_memory: bool) -> FullwaveSolver:
    work_dir = tmp_path
    (
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    ) = setup_inputs()

    solver = FullwaveSolver(
        work_dir=work_dir,
        simulation_params=simulation_params,
        domain_organizer=domain_organizer,
        transducer=transducer,
        wave_transmitter=wave_transmitter,
        signal_receiver=signal_receiver,
        initial_condition=initial_condition,
        on_memory=on_memory,
    )
    return solver


# parametrize the pytest with on_memory = True/False
@pytest.mark.parametrize("on_memory", [True, False])
def test_instance(tmp_path: Path, on_memory: bool):
    solver = build_full_instance(tmp_path, on_memory)
    assert solver.work_dir.exists()
    if on_memory:
        # check if the work_dir is on memory
        assert solver.work_dir.parent == Path("/dev/shm")


def test_save_data_for_beamforming(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    solver = build_full_instance(tmp_path, on_memory=False)
    solver._save_data_for_fsa_beamforming()

    workspace_mat_file_path = solver.work_dir / "launcher_workspace.mat"
    ground_truth_mat_file_path = test_data_dir / "launcher_workspace.mat"

    assert workspace_mat_file_path.exists()
    assert ground_truth_mat_file_path.exists()

    target_variable_in_xdc_name_list = [
        "npx",
        "nOutPx",
        "outcoords",
        "outcoords2",
        "incoords2",
        "rad",
        "cen",
    ]

    target_variable_in_tx_name_list = [
        "nTx",
        "bw",
        "fnumber",
    ]

    target_variable_in_maps_name_list = [
        "cmap",
    ]

    target_variable_name_list = [
        "fs",
        "f0",
        "c0",
        "dX",
        "dY",
        "lambda",
        "nY",
        "nX",
        "dT",
        "modT",
    ]

    test_variable_dict = scipy.io.loadmat(workspace_mat_file_path)
    ground_truth_variable_dict = scipy.io.loadmat(ground_truth_mat_file_path)

    for target_variable_name in target_variable_in_xdc_name_list:
        test_variable = test_variable_dict[target_variable_name]
        ground_truth_variable = ground_truth_variable_dict["xdc"][target_variable_name][0, 0]
        if not np.allclose(test_variable, ground_truth_variable, rtol=1e-06):
            raise ValueError(
                f"test_value {test_variable} and ground_truth {ground_truth_variable} are not close"
            )

    for target_variable_name in target_variable_in_tx_name_list:
        test_variable = test_variable_dict[target_variable_name]
        ground_truth_variable = ground_truth_variable_dict["tx"][target_variable_name][0, 0]
        if not np.allclose(test_variable, ground_truth_variable, rtol=1e-06):
            raise ValueError(
                f"test_value {test_variable} and ground_truth {ground_truth_variable} are not close"
            )

    for target_variable_name in target_variable_in_maps_name_list:
        test_variable = test_variable_dict[target_variable_name]
        ground_truth_variable = ground_truth_variable_dict["maps"][target_variable_name][0, 0]
        if not np.allclose(test_variable, ground_truth_variable, rtol=1e-06):
            raise ValueError(
                f"test_value {test_variable} and ground_truth {ground_truth_variable} are not close"
            )

    for target_variable_name in target_variable_name_list:
        test_variable = test_variable_dict[target_variable_name]
        ground_truth_variable = ground_truth_variable_dict[target_variable_name]
        if not np.allclose(test_variable, ground_truth_variable, rtol=1e-06):
            raise ValueError(
                f"test_value {test_variable} and ground_truth {ground_truth_variable} are not close"
            )
