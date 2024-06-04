from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fullwave_simulation
from fullwave_simulation.conditions import FSAInitialCondition
from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains import (
    Background,
    DomainOrganizer,
    Phantom,
    Scatterer,
    WaterGel,
    geometry_utils,
)
from fullwave_simulation.solvers import FSAInputGenerator
from fullwave_simulation.transducers import (
    C52VTransducer,
    ConvexTxWaveTransmitter,
    SignalReceiver,
)
from fullwave_simulation.utils import test_utils
from fullwave_simulation.utils.utils import matlab_round


class FSAInputGeneratorMod(FSAInputGenerator):
    def __init__(
        self,
        work_dir,
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition: FSAInitialCondition,
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
        self._c_map = domain_organizer.constructed_domain_dict["c_map"]
        self._rho_map = domain_organizer.constructed_domain_dict["rho_map"]
        self._A_map = domain_organizer.constructed_domain_dict["a_map"]
        self._beta_map = domain_organizer.constructed_domain_dict["beta_map"]
        self._input_coords_zero = geometry_utils.map_to_coordinates_matlab(
            domain_organizer.constructed_domain_dict["air_map"]
        )

        self._input_coords = self.transducer.incoords
        self._output_coords = self.transducer.outcoords

        self._r = self.simulation_params.cfl


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


def build_full_instance(tmp_path: Path):
    work_dir = tmp_path
    (
        simulation_params,
        domain_organizer,
        c52v_transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    ) = setup_inputs()

    input_generator = FSAInputGenerator(
        work_dir,
        simulation_params,
        domain_organizer,
        c52v_transducer,
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
        c52v_transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    ) = setup_inputs()

    input_generator = FSAInputGeneratorMod(
        work_dir,
        simulation_params,
        domain_organizer,
        c52v_transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
    )
    return input_generator


def test_full_instance(tmp_path: Path):
    input_generator = build_full_instance(tmp_path)
    assert isinstance(input_generator, FSAInputGenerator)


def test_mod_instance(tmp_path: Path):
    input_generator = build_init_modified_instance(tmp_path)
    assert isinstance(input_generator, FSAInputGenerator)


def test_set_field_params(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_field_params()
    nX = input_generator._nX
    nY = input_generator._nY
    nXe = input_generator._nXe
    nYe = input_generator._nYe
    nT = input_generator._nT

    nTic = input_generator._nTic
    for value, var_name in [
        [nX, "nX"],
        [nY, "nY"],
        [nXe, "nXe"],
        [nYe, "nYe"],
        [nT, "nT"],
        [nTic, "nTic"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
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
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_extend_input_maps(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    c, rho, beta, A, K = input_generator._extend_input_maps(
        c_map=input_generator._c_map,
        rho_map=input_generator._rho_map,
        beta_map=input_generator._beta_map,
        A_map=input_generator._A_map,
    )

    for value, var_name in [
        [c, "c"],
        [rho, "rho"],
        [beta, "beta"],
        [A, "A"],
        [K, "K"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_set_coords(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_field_params()
    input_generator._set_step_params()
    (
        input_generator._c,
        input_generator._rho,
        input_generator._beta,
        input_generator._A,
        input_generator._K,
    ) = input_generator._extend_input_maps(
        c_map=input_generator._c_map,
        rho_map=input_generator._rho_map,
        beta_map=input_generator._beta_map,
        A_map=input_generator._A_map,
    )
    input_generator._dim = int(
        matlab_round(input_generator._c.max()) - matlab_round(input_generator._c.min())
    )

    input_generator._set_coords()
    n_coords = input_generator._n_coords
    n_coords_out = input_generator._n_coords_out
    input_coords = input_generator._input_coords
    output_coords = input_generator._output_coords

    for value, var_name in [
        [n_coords, "ncoords"],
        [n_coords_out, "ncoordsout"],
        [input_coords, "incoords"],
        [output_coords, "outcoords"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_set_d_mat(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_d_mat()
    d = input_generator._d

    for value, var_name in [
        [d, "d"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_set_d_map(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_field_params()
    input_generator._set_step_params()
    (
        input_generator._c,
        input_generator._rho,
        input_generator._beta,
        input_generator._A,
        input_generator._K,
    ) = input_generator._extend_input_maps(
        c_map=input_generator._c_map,
        rho_map=input_generator._rho_map,
        beta_map=input_generator._beta_map,
        A_map=input_generator._A_map,
    )
    input_generator._dim = int(
        matlab_round(input_generator._c.max()) - matlab_round(input_generator._c.min())
    )

    input_generator._set_d_map(input_generator._dim, input_generator._c)
    d_map = input_generator._d_map

    for value, var_name in [
        [d_map, "dmap"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_set_dc_map(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_field_params()
    input_generator._set_step_params()
    (
        input_generator._c,
        input_generator._rho,
        input_generator._beta,
        input_generator._A,
        input_generator._K,
    ) = input_generator._extend_input_maps(
        c_map=input_generator._c_map,
        rho_map=input_generator._rho_map,
        beta_map=input_generator._beta_map,
        A_map=input_generator._A_map,
    )
    input_generator._dim = int(
        matlab_round(input_generator._c.max()) - matlab_round(input_generator._c.min())
    )

    input_generator._set_dc_map(input_generator._c)
    dc_map = input_generator._dc_map

    for value, var_name in [
        [dc_map, "dcmap"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_set_pmls_params(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_init_modified_instance(tmp_path)
    input_generator._set_step_params()
    input_generator._set_pmls_params()
    L = input_generator._L
    d0 = input_generator._d0

    for value, var_name in [
        [L, "L"],
        [d0, "d0"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_calc_kappa(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_full_instance(tmp_path)
    x_start = input_generator._obtain_x_start_attenuation_dispersion_curves()
    kappa_x, kappa_y, kappa_u, kappa_w = input_generator._calc_kappa(x_start, A=input_generator._A)

    for value, var_name in [
        [kappa_x, "kappax"],
        [kappa_y, "kappay"],
        [kappa_u, "kappau"],
        [kappa_w, "kappaw"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_calc_pml_boundary_conditions(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_full_instance(tmp_path)
    x_start = input_generator._obtain_x_start_attenuation_dispersion_curves()
    (
        a_pml_u1,
        b_pml_u1,
        a_pml_x1,
        b_pml_x1,
        a_pml_x2,
        b_pml_x2,
        a_pml_u2,
        b_pml_u2,
        #
        a_pml_x_old,
        a_pml_x_old2,
        a_pml_y_old,
        a_pml_y_old2,
        b_pml_x_old,
        b_pml_x_old2,
        b_pml_y_old,
        b_pml_y_old2,
    ) = input_generator._calc_pml_boundary_conditions(x_start, A=input_generator._A)

    for value, var_name in [
        [a_pml_u1, "apmlu1"],
        [b_pml_u1, "bpmlu1"],
        [a_pml_x1, "apmlx1"],
        [b_pml_x1, "bpmlx1"],
        [a_pml_x2, "apmlx2"],
        [b_pml_x2, "bpmlx2"],
        [a_pml_u2, "apmlu2"],
        [b_pml_u2, "bpmlu2"],
        #
        [a_pml_x_old, "apmlxOld"],
        [a_pml_x_old2, "apmlxOld2"],
        [a_pml_y_old, "apmlyOld"],
        [a_pml_y_old2, "apmlyOld2"],
        [b_pml_x_old, "bpmlxOld"],
        [b_pml_x_old2, "bpmlxOld2"],
        [b_pml_y_old, "bpmlyOld"],
        [b_pml_y_old2, "bpmlyOld2"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_pml_{var_name.lower()}.mat",
            var_name=var_name,
            test_value=value,
        )


def test_calc_gradient_masks(tmp_path: Path):
    test_data_dir = test_utils.get_test_data_dir("solvers")
    input_generator = build_full_instance(tmp_path)
    x_start = input_generator._obtain_x_start_attenuation_dispersion_curves()

    (
        a_pml_u1,
        b_pml_u1,
        a_pml_x1,
        b_pml_x1,
        a_pml_x2,
        b_pml_x2,
        a_pml_u2,
        b_pml_u2,
        #
        a_pml_x_old,
        a_pml_x_old2,
        a_pml_y_old,
        a_pml_y_old2,
        b_pml_x_old,
        b_pml_x_old2,
        b_pml_y_old,
        b_pml_y_old2,
    ) = input_generator._calc_pml_boundary_conditions(x_start, input_generator._A)

    (
        a_pml_u1,
        b_pml_u1,
        a_pml_w1,
        b_pml_w1,
        #
        a_pml_x1,
        b_pml_x1,
        a_pml_y1,
        b_pml_y1,
        #
        a_pml_u2,
        b_pml_u2,
        a_pml_w2,
        b_pml_w2,
        #
        a_pml_x2,
        b_pml_x2,
        a_pml_y2,
        b_pml_y2,
    ) = input_generator._calc_gradient_masks(
        a_pml_u1,
        b_pml_u1,
        a_pml_x1,
        b_pml_x1,
        a_pml_x2,
        b_pml_x2,
        a_pml_u2,
        b_pml_u2,
        #
        a_pml_x_old,
        a_pml_x_old2,
        a_pml_y_old,
        a_pml_y_old2,
        b_pml_x_old,
        b_pml_x_old2,
        b_pml_y_old,
        b_pml_y_old2,
    )

    for value, var_name in [
        [a_pml_u1, "apmlu1"],
        [b_pml_u1, "bpmlu1"],
        [a_pml_w1, "apmlw1"],
        [b_pml_w1, "bpmlw1"],
        #
        [a_pml_x1, "apmlx1"],
        [b_pml_x1, "bpmlx1"],
        [a_pml_y1, "apmly1"],
        [b_pml_y1, "bpmly1"],
        #
        [a_pml_u2, "apmlu2"],
        [b_pml_u2, "bpmlu2"],
        [a_pml_w2, "apmlw2"],
        [b_pml_w2, "bpmlw2"],
        #
        [a_pml_x2, "apmlx2"],
        [b_pml_x2, "bpmlx2"],
        [a_pml_y2, "apmly2"],
        [b_pml_y2, "bpmly2"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"input_gen_grad_{var_name}.mat",
            var_name=var_name,
            test_value=value,
        )


@dataclass
class DataForInputGenerator:
    pass


def test_save_variables_into_dat_file(tmp_path):
    # input fixed result and compare with the matlab result
    # read the exported files and compare with the matlab result
    test_data_dir = test_utils.get_test_data_dir("solvers")
    conditions_test_data_dir = test_utils.get_test_data_dir("conditions")

    input_generator = build_full_instance(tmp_path)
    simulation_dir = tmp_path / "1"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    data_for_input_generator = DataForInputGenerator()

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
    for var_name in kappa_name_list:
        var = test_utils.load_test_variable(
            mat_file_path=test_data_dir / f"input_gen_{var_name}.mat",
            var_name=var_name,
        )
        setattr(data_for_input_generator, var_name, var)

    for var_name in pml_name_list:
        var = test_utils.load_test_variable(
            mat_file_path=test_data_dir / f"input_gen_grad_{var_name}.mat",
            var_name=var_name,
        )
        setattr(data_for_input_generator, var_name, var)

    initial_condition_mat = test_utils.load_test_variable(
        mat_file_path=conditions_test_data_dir / "icmat.mat",
        var_name="icmat",
    )
    setattr(data_for_input_generator, "initial_condition_mat", initial_condition_mat)

    input_generator._save_variables_into_dat_file(
        simulation_dir=simulation_dir,
        kappa_x=data_for_input_generator.kappax,
        kappa_y=data_for_input_generator.kappay,
        kappa_u=data_for_input_generator.kappau,
        kappa_w=data_for_input_generator.kappaw,
        a_pml_u1=data_for_input_generator.apmlu1,
        b_pml_u1=data_for_input_generator.bpmlu1,
        a_pml_w1=data_for_input_generator.apmlw1,
        b_pml_w1=data_for_input_generator.bpmlw1,
        a_pml_x1=data_for_input_generator.apmlx1,
        b_pml_x1=data_for_input_generator.bpmlx1,
        a_pml_y1=data_for_input_generator.apmly1,
        b_pml_y1=data_for_input_generator.bpmly1,
        a_pml_u2=data_for_input_generator.apmlu2,
        b_pml_u2=data_for_input_generator.bpmlu2,
        a_pml_w2=data_for_input_generator.apmlw2,
        b_pml_w2=data_for_input_generator.bpmlw2,
        a_pml_x2=data_for_input_generator.apmlx2,
        b_pml_x2=data_for_input_generator.bpmlx2,
        a_pml_y2=data_for_input_generator.apmly2,
        b_pml_y2=data_for_input_generator.bpmly2,
        dim=input_generator._dim,
        # initial_condition_mat=data_for_input_generator.initial_condition_mat,
    )

    variable_name_list = kappa_name_list + pml_name_list

    for var_name in variable_name_list:
        dat_file_path = simulation_dir / f"{var_name}.dat"
        test_dat_file_path = test_data_dir / "test_event_1" / f"{var_name}.dat"
        test_utils.load_and_check_dat_data(
            dat_file_path,
            test_dat_file_path,
            dtype=np.float32,
        )

    # dat_file_path = simulation_dir / "icmat.dat"
    # test_dat_file_path = test_data_dir / "test_event_1" / "icmat.dat"
    # test_utils.load_and_check_dat_data(
    #     dat_file_path,
    #     test_dat_file_path,
    #     dtype=np.float32,
    # )


def test_run(tmp_path):
    # integration test
    # read the exported files and compare with the matlab result
    test_data_dir = test_utils.get_test_data_dir("solvers")

    for i_event in [0, 9, 63, 127]:
        simulation_dir = tmp_path / f"{i_event + 1}"

        input_generator = build_full_instance(tmp_path)
        input_generator.run(i_event=i_event)

        event_dependent_list = ["icmat"]
        coords_name_list = ["icc", "outc"]
        map_name_list = ["c", "K", "rho", "beta"]
        step_name_list = ["dX", "dY", "dT", "c0"]
        coords_param_list = ["nX", "nY", "nT", "ncoords", "ncoordsout", "nTic", "modT"]
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
            + step_name_list
            + coords_param_list
            + d_param_list
            + kappa_name_list
            + pml_name_list
        )

        for var_name in variable_name_list:
            dat_file_path = simulation_dir.parent / f"{var_name}.dat"
            test_dat_file_path = test_data_dir / "test_event_1" / f"{var_name}.dat"
            test_utils.load_and_check_dat_data(
                dat_file_path,
                test_dat_file_path,
                dtype=np.float32,
            )

        for var_name in coords_name_list + int_d_param_list:
            dat_file_path = simulation_dir.parent / f"{var_name}.dat"
            test_dat_file_path = test_data_dir / "test_event_1" / f"{var_name}.dat"
            test_utils.load_and_check_dat_data(
                dat_file_path,
                test_dat_file_path,
                dtype=np.int32,
            )

        for var_name in event_dependent_list:
            dat_file_path = simulation_dir / f"{var_name}.dat"
            test_dat_file_path = test_data_dir / f"test_event_{i_event + 1}" / f"{var_name}.dat"
            test_utils.load_and_check_dat_data(
                dat_file_path,
                test_dat_file_path,
                dtype=np.float32,
            )
