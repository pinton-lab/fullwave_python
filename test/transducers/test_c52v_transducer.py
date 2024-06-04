from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains import geometry_utils
from fullwave_simulation.transducers import C52VTransducer, Transducer
from fullwave_simulation.transducers.covex_receiver_map import ConvexReceiverMap
from fullwave_simulation.transducers.covex_transmitter_map import ConvexTransmitterMap
from fullwave_simulation.utils import test_utils

# pylint: disable=protected-access


class C52VTransducerMod(C52VTransducer):
    def __init__(
        self,
        simulation_params,
        material_properties,
        rho0=None,
        beta0=None,
        c0=None,
        a0=None,
        element_pitch=0.000508,
    ):
        self.simulation_params = simulation_params
        self.material_properties = material_properties
        self.lambda_ = simulation_params.lambda_
        self.freq_div = simulation_params.freq_div
        self.wX = simulation_params.wX
        self.wY = simulation_params.wY
        self.rho0 = rho0
        self.beta0 = beta0
        self.c0 = c0
        self.a0 = a0
        self.element_pitch = element_pitch


def build_full_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()
    c52v_transducer = C52VTransducer(simulation_params, material_properties)
    return c52v_transducer


def build_init_modified_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()
    c52v_transducer = C52VTransducerMod(simulation_params, material_properties)
    return c52v_transducer


def test_full_instance():
    c52v_transducer = build_full_instance()
    assert isinstance(c52v_transducer, Transducer)
    assert isinstance(c52v_transducer, C52VTransducer)


def test_properties():
    transducer = build_full_instance()
    isinstance(transducer.simulation_params, FSASimulationParams)
    isinstance(transducer.material_properties, MaterialProperties)


def test_define_grid_params_based_on_ppw():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transducer = build_init_modified_instance()
    transducer._define_grid_params_based_on_ppw()

    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_ppw.mat",
        var_name="ppw",
        test_value=transducer.ppw,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_nX.mat",
        var_name="nX",
        test_value=transducer.num_x,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_nY.mat",
        var_name="nY",
        test_value=transducer.num_y,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_dX.mat",
        var_name="dX",
        test_value=transducer.dX,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_dY.mat",
        var_name="dY",
        test_value=transducer.dY,
    )


def test_compute_transducer_variables():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transducer = build_init_modified_instance()
    transducer._define_grid_params_based_on_ppw()
    transducer._compute_transducer_variables()

    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_rad.mat",
        var_name="rad",
        test_value=transducer.rad,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_npx.mat",
        var_name="npx",
        test_value=transducer.npx,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_ptch.mat",
        var_name="ptch",
        test_value=transducer.subelem_ptch,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_dTheta.mat",
        var_name="dTheta",
        test_value=transducer.dTheta,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_cen.mat",
        var_name="cen",
        test_value=transducer.cen,
    )


def test_define_theta_at_center_of_elements():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transducer = build_init_modified_instance()
    dTheta = float(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "c52v_dTheta.mat", var_name="dTheta"
        )[0, 0]
    )
    npx = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "c52v_npx.mat", var_name="npx"
        )[0, 0]
    )

    thetas = transducer._define_theta_at_center_of_elements(dTheta, npx)

    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_thetas.mat",
        var_name="thetas",
        test_value=thetas,
    )


def test_make_transducer_surface_map():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transducer = build_init_modified_instance()

    transducer.ppw = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_ppw.mat",
        var_name="ppw",
    )[0, 0]

    num_x = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "c52v_nX.mat",
            var_name="nX",
        )[0, 0]
    )

    num_y = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "c52v_nY.mat",
            var_name="nY",
        )[0, 0]
    )

    cen = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_cen.mat",
        var_name="cen",
    )[0]

    rad = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_rad.mat",
        var_name="rad",
    )[0, 0]

    convex_transmitter_map, convex_receiver_map = transducer._make_transducer_surface_map(
        nX=num_x,
        nY=num_y,
        cen=cen,
        rad=rad,
    )

    isinstance(convex_transmitter_map, ConvexTransmitterMap)
    isinstance(convex_receiver_map, ConvexReceiverMap)

    in_map = convex_transmitter_map.in_map
    out_map = convex_receiver_map.out_map

    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_inmap.mat",
        var_name="inmap",
        test_value=in_map,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_outmap.mat",
        var_name="outmap",
        test_value=out_map,
    )

    incoords = geometry_utils.map_to_coords_with_sort(in_map)
    outcoords = geometry_utils.map_to_coords_with_sort(out_map)
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_incoords_sorted.mat",
        var_name="incoords",
        test_value=incoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_outcoords_sorted.mat",
        var_name="outcoords",
        test_value=outcoords,
    )


def test_assign_transducer_num_to_input():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transducer = build_init_modified_instance()

    incoords = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_incoords_sorted.mat",
        var_name="incoords",
    )
    outcoords = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_outcoords_sorted.mat",
        var_name="outcoords",
    )
    dTheta = float(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "c52v_dTheta.mat", var_name="dTheta"
        )[0, 0]
    )
    npx = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "c52v_npx.mat", var_name="npx"
        )[0, 0]
    )
    cen = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_cen.mat",
        var_name="cen",
    )[0]

    thetas = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_thetas.mat",
        var_name="thetas",
    )[0]

    (
        thetas_in,
        thetas_out,
        incoords,
        outcoords,
        incoords2,
        outcoords2,
        nOutPx,
        nInPx,
    ) = transducer._assign_transducer_num_to_input(
        incoords=incoords,
        outcoords=outcoords,
        cen=cen,
        npx=npx,
        dTheta=dTheta,
        thetas=thetas,
    )

    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_thetas_in.mat",
        var_name="thetas_in",
        test_value=thetas_in,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_thetas_out.mat",
        var_name="thetas_out",
        test_value=thetas_out,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_incoords_assigned.mat",
        var_name="incoords",
        test_value=incoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_outcoords_assigned.mat",
        var_name="outcoords",
        test_value=outcoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_incoords2.mat",
        var_name="incoords2",
        test_value=incoords2,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_outcoords2.mat",
        var_name="outcoords2",
        test_value=outcoords2,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_nOutPx.mat",
        var_name="nOutPx",
        test_value=nOutPx,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_nInPx.mat",
        var_name="nInPx",
        test_value=nInPx,
    )


def test_setup():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transducer = build_init_modified_instance()
    transducer._define_grid_params_based_on_ppw()
    transducer._compute_transducer_variables()
    transducer._setup()

    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_thetas_in.mat",
        var_name="thetas_in",
        test_value=transducer.thetas_in,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_thetas_out.mat",
        var_name="thetas_out",
        test_value=transducer.thetas_out,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_incoords_assigned.mat",
        var_name="incoords",
        test_value=transducer.incoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_outcoords_assigned.mat",
        var_name="outcoords",
        test_value=transducer.outcoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_incoords2.mat",
        var_name="incoords2",
        test_value=transducer.incoords2,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_outcoords2.mat",
        var_name="outcoords2",
        test_value=transducer.outcoords2,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_nOutPx.mat",
        var_name="nOutPx",
        test_value=transducer.nOutPx,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_nInPx.mat",
        var_name="nInPx",
        test_value=transducer.nInPx,
    )


def test_full():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transducer = build_full_instance()
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_thetas_in.mat",
        var_name="thetas_in",
        test_value=transducer.thetas_in,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_thetas_out.mat",
        var_name="thetas_out",
        test_value=transducer.thetas_out,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_incoords_assigned.mat",
        var_name="incoords",
        test_value=transducer.incoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_outcoords_assigned.mat",
        var_name="outcoords",
        test_value=transducer.outcoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_incoords2.mat",
        var_name="incoords2",
        test_value=transducer.incoords2,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_outcoords2.mat",
        var_name="outcoords2",
        test_value=transducer.outcoords2,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_nOutPx.mat",
        var_name="nOutPx",
        test_value=transducer.nOutPx,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_nInPx.mat",
        var_name="nInPx",
        test_value=transducer.nInPx,
    )
