from pathlib import Path

from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.transducers.covex_transmitter_map import ConvexTransmitterMap
from fullwave_simulation.utils import test_utils


class ConvexTransmitterMapMod(ConvexTransmitterMap):
    def __init__(
        self,
        cen,
        rad,
        num_x: int,
        num_y: int,
        ppw,
        material_properties,
        simulation_params,
        name="transmitter",
    ):
        self.num_x = num_x
        self.num_y = num_y
        self.material_properties = material_properties
        self.simulation_params = simulation_params
        self.name = name

        self.cen = cen
        self.rad = rad
        self.ppw = ppw


def build_full_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    test_data_dir = test_utils.get_test_data_dir("transducers")
    ppw = test_utils.load_test_variable(
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
    convex_transmitter_map = ConvexTransmitterMap(
        cen=cen,
        rad=rad,
        num_x=num_x,
        num_y=num_y,
        ppw=ppw,
        material_properties=material_properties,
        simulation_params=simulation_params,
    )
    return convex_transmitter_map


def build_init_modified_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    test_data_dir = test_utils.get_test_data_dir("transducers")
    ppw = test_utils.load_test_variable(
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
    convex_transmitter_map = ConvexTransmitterMapMod(
        cen=cen,
        rad=rad,
        num_x=num_x,
        num_y=num_y,
        ppw=ppw,
        material_properties=material_properties,
        simulation_params=simulation_params,
    )
    return convex_transmitter_map


def test_instance():
    convex_transmitter_map = build_full_instance()
    assert isinstance(convex_transmitter_map, ConvexTransmitterMap)


def test_calculate_inmap():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    convex_transmitter_map = build_init_modified_instance()
    in_map = convex_transmitter_map._calculate_inmap()

    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_inmap.mat",
        var_name="inmap",
        test_value=in_map,
    )


def test_make_transducer_surface_label():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    convex_transmitter_map = build_init_modified_instance()
    ppw = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_ppw.mat",
        var_name="ppw",
    )[0, 0]
    num_x = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "c52v_nX.mat",
            var_name="nX",
        )[0, 0]
    )

    in_map = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "c52v_inmap.mat",
        var_name="inmap",
    )

    surface_label = convex_transmitter_map._make_transducer_surface_label(
        num_x=num_x,
        inmap=in_map,
        ppw=ppw,
    )

    test_utils.check_variable(
        mat_file_path=test_data_dir / "c52v_surf.mat",
        var_name="surf",
        test_value=surface_label,
    )
