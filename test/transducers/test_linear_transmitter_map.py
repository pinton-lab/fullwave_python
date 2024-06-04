from pathlib import Path

from fullwave_simulation.constants import FocusedSimulationParams, MaterialProperties
from fullwave_simulation.transducers import LinearTransmitterMap
from fullwave_simulation.utils import test_utils


class LinearTransmitterMapMod(LinearTransmitterMap):
    def __init__(
        self,
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
        self.ppw = ppw


def build_full_instance():
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialProperties()

    test_data_dir = test_utils.get_test_data_dir("transducers")
    ppw = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "l125_ppw.mat",
        var_name="ppw",
    )[0, 0]
    num_x = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "l125_nX.mat",
            var_name="nX",
        )[0, 0]
    )

    num_y = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "l125_nY.mat",
            var_name="nY",
        )[0, 0]
    )
    convex_transmitter_map = LinearTransmitterMap(
        num_x=num_x,
        num_y=num_y,
        ppw=ppw,
        material_properties=material_properties,
        simulation_params=simulation_params,
    )
    return convex_transmitter_map


def build_init_modified_instance():
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialProperties()

    test_data_dir = test_utils.get_test_data_dir("transducers")
    ppw = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "l125_ppw.mat",
        var_name="ppw",
    )[0, 0]
    num_x = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "l125_nX.mat",
            var_name="nX",
        )[0, 0]
    )

    num_y = int(
        test_utils.load_test_variable(
            mat_file_path=test_data_dir / "l125_nY.mat",
            var_name="nY",
        )[0, 0]
    )

    transmitter_map = LinearTransmitterMapMod(
        num_x=num_x,
        num_y=num_y,
        ppw=ppw,
        material_properties=material_properties,
        simulation_params=simulation_params,
    )
    return transmitter_map


def test_instance():
    transmitter_map = build_full_instance()
    assert isinstance(transmitter_map, LinearTransmitterMap)


def test_calculate_inmap():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transmitter_map = build_init_modified_instance()
    in_map = transmitter_map._calculate_inmap()

    test_utils.check_variable(
        mat_file_path=test_data_dir / "l125_inmap.mat",
        var_name="inmap",
        test_value=in_map,
    )
