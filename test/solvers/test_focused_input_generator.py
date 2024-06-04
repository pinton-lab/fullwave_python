from pathlib import Path

import numpy as np
import pytest

import fullwave_simulation
from fullwave_simulation.conditions import FocusedInitialCondition
from fullwave_simulation.constants import Constant, FocusedSimulationParams
from fullwave_simulation.domains import AbdominalWall, Background, DomainOrganizer
from fullwave_simulation.solvers import FocusedInputGenerator
from fullwave_simulation.transducers import (
    L125Transducer,
    LinearTxWaveTransmitter,
    SignalReceiver,
)
from fullwave_simulation.utils import test_utils
from fullwave_simulation.utils.utils import matlab_round


class MaterialPropertiesFocused(Constant):
    fat = {"bovera": 9.6, "alpha": 0.48, "ppower": 1.1, "c0": 1478, "rho0": 950}
    fat["beta"] = 1 + fat["bovera"] / 2

    # liver = {"bovera": 7.6, "alpha": 0.5, "ppower": 1.1, "c0": 1570, "rho0": 1064}
    # liver["beta"] = 1 + liver["bovera"] / 2

    muscle = {"bovera": 9, "alpha": 1.09, "ppower": 1.0, "c0": 1547, "rho0": 1050}
    muscle["beta"] = 1 + muscle["bovera"] / 2

    water = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1480, "rho0": 1000}
    water["beta"] = 1 + water["bovera"] / 2

    skin = {"bovera": 8, "alpha": 2.1, "ppower": 1, "c0": 1498, "rho0": 1000}
    skin["beta"] = 1 + skin["bovera"] / 2

    tissue = {"bovera": 9, "alpha": 0.5, "ppower": 1, "c0": 1540, "rho0": 1000}
    tissue["beta"] = 1 + tissue["bovera"] / 2

    connective = {"bovera": 8, "alpha": 1.57, "ppower": 1, "c0": 1613, "rho0": 1120}
    connective["beta"] = 1 + connective["bovera"] / 2

    blood = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1520, "rho0": 1000}
    blood["beta"] = 1 + blood["bovera"] / 2

    lung_fluid = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1440, "rho0": 1000}
    lung_fluid["beta"] = 1 + lung_fluid["bovera"] / 2

    c0 = 1540
    rho0 = 1000
    a0 = 0.5
    beta0 = 0


class FocusedInputGeneratorMod(FocusedInputGenerator):
    def __init__(
        self,
        work_dir,
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
        path_fullwave_simulation_bin=(
            Path(fullwave_simulation.__file__).parent.parent
            / "fullwave_simulation/bin/fullwave2_try6_nln_relaxing_pzero_rebuild3"
        ),
        m=8,
        num_body=40,
    ):
        self._work_dir = Path(work_dir)
        self._path_fullwave_simulation_bin = path_fullwave_simulation_bin
        assert self._path_fullwave_simulation_bin.exists()

        self.simulation_params = simulation_params
        self.domain_organizer = domain_organizer

        self.transducer = transducer
        self.wave_transmitter = wave_transmitter
        self.signal_receiver = signal_receiver

        self.initial_condition = initial_condition

        self._m = m
        self._num_body = num_body
        self._c0 = self.simulation_params.c0
        self._omega0 = self.simulation_params.omega0
        self._duration = self.simulation_params.dur
        self._ppw = self.transducer.ppw
        self._cfl = self.simulation_params.cfl
        self._modT = self.simulation_params.modT
        self._c_map = domain_organizer.constructed_domain_dict["c_map"]
        self._rho_map = domain_organizer.constructed_domain_dict["rho_map"]
        self._A_map = domain_organizer.constructed_domain_dict["a_map"]
        self._beta_map = domain_organizer.constructed_domain_dict["beta_map"]

        self._input_coords = self.transducer.incoords
        self._output_coords = self.transducer.outcoords

        self._r = self.simulation_params.cfl
        # init modified to delete the following line


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
        apply_tissue_deformation=False,
        apply_tissue_compression=True,
        use_smoothing=True,
        skip_i0=False,
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


def build_full_instance(tmp_path: Path):
    work_dir = tmp_path
    (
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    ) = setup_inputs_linear_tx_focused()

    input_generator = FocusedInputGenerator(
        work_dir,
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    )
    return input_generator


def build_init_modified_instance(tmp_path: Path):
    work_dir = tmp_path
    (
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    ) = setup_inputs_linear_tx_focused()

    input_generator = FocusedInputGeneratorMod(
        work_dir,
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    )
    return input_generator


def test_full_instance(tmp_path: Path):
    input_generator = build_full_instance(tmp_path)
    assert isinstance(input_generator, FocusedInputGenerator)


def test_mod_instance(tmp_path: Path):
    input_generator = build_init_modified_instance(tmp_path)
    assert isinstance(input_generator, FocusedInputGenerator)


def test_set_field_params(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_field_params()
    nX = input_generator._nX
    nY = input_generator._nY
    nT = input_generator._nT
    for value, var_name in [
        [nX, "nX"],
        [nY, "nY"],
        [nT, "nT"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_l125_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_set_step_params(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_step_params()
    dX = input_generator._dX
    dY = input_generator._dY
    dT = input_generator._dT
    for value, var_name in [
        [dX, "dX"],
        [dY, "dY"],
        [dT, "dT"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_l125_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_set_coords(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_field_params()
    input_generator._set_step_params()

    input_generator._set_coords()
    # n_coords = input_generator._n_coords
    # n_coords_out = input_generator._n_coords_out
    input_coords = input_generator._input_coords_org
    output_coords = input_generator._output_coords_org

    for value, var_name in [
        # [n_coords, "ncoords"],
        # [n_coords_out, "ncoordsout"],
        [input_coords, "incoords"],
        [output_coords, "outcoords"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_l125_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


@pytest.mark.parametrize("i_event", [0, 63])
def test_cut_domain_for_scan(tmp_path, i_event):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    event_test_dir = test_data_dir / f"linear_tx_test_event_{i_event+1}"
    simulation_dir = tmp_path / f"{i_event + 1}"
    input_generator = build_full_instance(tmp_path)

    # --- run sequentially ---
    simulation_dir = input_generator._work_dir / f"{i_event + 1}"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    c, rho, beta, A, _, orig = input_generator._cut_domain_for_scan(i_event)

    test_utils.check_variable(
        mat_file_path=event_test_dir / "orig.mat",
        var_name="orig_test",
        test_value=orig + 1,
    )

    for value, var_name in [
        [c, "cmap"],
        [rho, "rhomap"],
        [A, "Amap"],
        [beta, "betamap"],
    ]:
        test_utils.check_variable(
            mat_file_path=event_test_dir / f"{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


@pytest.mark.parametrize("i_event", [0, 63])
def test_extend_input_maps(tmp_path, i_event):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    event_test_dir = test_data_dir / f"linear_tx_test_event_{i_event+1}"
    simulation_dir = tmp_path / f"{i_event + 1}"
    input_generator = build_full_instance(tmp_path)

    # --- run sequentially ---
    simulation_dir = input_generator._work_dir / f"{i_event + 1}"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    c, rho, beta, A, _, _ = input_generator._cut_domain_for_scan(i_event)
    c, rho, beta, A, K = input_generator._extend_input_maps(
        c_map=c, rho_map=rho, beta_map=beta, A_map=A
    )
    for value, var_name in [
        [c, "c"],
        [rho, "rho"],
        [A, "A"],
        [beta, "beta"],
    ]:
        test_utils.check_variable(
            mat_file_path=event_test_dir / f"{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


@pytest.mark.parametrize("i_event", [0, 63])
def test_set_d_map(tmp_path, i_event):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    event_test_dir = test_data_dir / f"linear_tx_test_event_{i_event+1}"
    simulation_dir = tmp_path / f"{i_event + 1}"
    input_generator = build_full_instance(tmp_path)

    # --- run sequentially ---
    simulation_dir = input_generator._work_dir / f"{i_event + 1}"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    c, rho, beta, A, _, _ = input_generator._cut_domain_for_scan(i_event)
    c, rho, beta, A, K = input_generator._extend_input_maps(
        c_map=c, rho_map=rho, beta_map=beta, A_map=A
    )
    dim = int(matlab_round(c.max()) - matlab_round(c.min()))
    input_generator._set_d_map(dim, c)
    d_map = input_generator._d_map
    test_utils.check_variable(
        mat_file_path=event_test_dir / "dmap.mat",
        var_name="dmap",
        test_value=d_map,
    )


@pytest.mark.parametrize("i_event", [0, 63])
def test_calc_kappa(tmp_path: Path, i_event):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    event_test_dir = test_data_dir / f"linear_tx_test_event_{i_event+1}"
    simulation_dir = tmp_path / f"{i_event + 1}"
    input_generator = build_full_instance(tmp_path)

    # --- run sequentially ---
    simulation_dir = input_generator._work_dir / f"{i_event + 1}"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    initial_condition_mat = input_generator.initial_condition.generate_icmat()

    input_generator._write_ic(simulation_dir / "icmat.dat", np.transpose(initial_condition_mat))
    input_generator._copy_simulation_bin_file(simulation_dir)

    orig = int(
        np.round(input_generator._c_map.shape[0] / 2)
        - np.round(input_generator._nX / 2)
        - np.round(
            ((i_event + 1) - (input_generator.simulation_params.nevents + 1) / 2)
            * input_generator.transducer.beam_spacing
        )
        - 6
    )
    c = input_generator._c_map[orig : orig + input_generator._nX, 0 : input_generator._nY]
    rho = input_generator._rho_map[orig : orig + input_generator._nX, 0 : input_generator._nY]
    A = input_generator._A_map[orig : orig + input_generator._nX, 0 : input_generator._nY]
    beta = input_generator._beta_map[orig : orig + input_generator._nX, 0 : input_generator._nY]

    c, rho, beta, A, K = input_generator._extend_input_maps(
        c_map=c, rho_map=rho, beta_map=beta, A_map=A
    )
    input_generator._save_maps(simulation_dir=simulation_dir, c=c, K=K, rho=rho, beta=beta)
    dim = int(matlab_round(c.max()) - matlab_round(c.min()))
    input_generator._set_d_map(dim, c)
    input_generator._set_dc_map(c)

    x_start = input_generator._obtain_x_start_attenuation_dispersion_curves()

    kappa_x, kappa_y, kappa_u, kappa_w = input_generator._calc_kappa(x_start, A)

    for value, var_name in [
        [kappa_x, "kappax"],
        [kappa_y, "kappay"],
        [kappa_u, "kappau"],
        [kappa_w, "kappaw"],
    ]:
        test_utils.check_variable(
            mat_file_path=event_test_dir / f"{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_run(tmp_path):
    # integration test
    # read the exported files and compare with the matlab result
    test_data_dir = test_utils.get_test_data_dir("solvers")

    for i_event in [0, 63]:
        simulation_dir = tmp_path / f"{i_event + 1}"

        input_generator = build_full_instance(tmp_path)
        input_generator.run(i_event=i_event)

        event_dependent_list = ["icmat"]
        coords_name_list = ["icc", "outc"]
        map_name_list = ["c", "rho", "K", "beta"]
        step_name_list = ["dX", "dY", "dT", "c0"]
        # coords_param_list = ["nX", "nY", "nT", "ncoords", "ncoordsout", "nTic", "modT"]
        coords_param_list = ["nX", "nY", "nT", "ncoords", "ncoordsout", "nTic"]
        d_param_list = ["d", "dmap", "ndmap"]

        int_d_param_list = ["dcmap"]

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
            map_name_list
            + kappa_name_list
            + pml_name_list
            + step_name_list
            + coords_param_list
            + d_param_list
        )

        for var_name in variable_name_list:
            dat_file_path = simulation_dir / f"{var_name}.dat"
            test_dat_file_path = (
                test_data_dir / f"linear_tx_test_event_{i_event+1}" / f"{var_name}.dat"
            )
            test_utils.load_and_check_dat_data(
                dat_file_path,
                test_dat_file_path,
                dtype=np.float32,
                array_shape=(input_generator._nXe, input_generator._nYe),
                export_diff_image_when_false=True,
            )

        for var_name in coords_name_list + int_d_param_list:
            dat_file_path = simulation_dir / f"{var_name}.dat"
            test_dat_file_path = (
                test_data_dir / f"linear_tx_test_event_{i_event+1}" / f"{var_name}.dat"
            )
            test_utils.load_and_check_dat_data(
                dat_file_path,
                test_dat_file_path,
                dtype=np.int32,
            )

        for var_name in event_dependent_list:
            dat_file_path = simulation_dir / f"{var_name}.dat"
            test_dat_file_path = (
                test_data_dir / f"linear_tx_test_event_{i_event+1}" / f"{var_name}.dat"
            )
            test_utils.load_and_check_dat_data(
                dat_file_path,
                test_dat_file_path,
                dtype=np.float32,
                array_shape=(input_generator._nXe, input_generator._nYe),
                export_diff_image_when_false=True,
            )
