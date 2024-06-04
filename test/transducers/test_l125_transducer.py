from fullwave_simulation.constants import FocusedSimulationParams, MaterialProperties
from fullwave_simulation.domains import geometry_utils
from fullwave_simulation.transducers import L125Transducer, Transducer
from fullwave_simulation.transducers.linear_receiver_map import LinearReceiverMap
from fullwave_simulation.transducers.linear_transmitter_map import LinearTransmitterMap
from fullwave_simulation.utils import test_utils

# pylint: disable=protected-access


class L125TransducerMod(L125Transducer):
    def __init__(
        self,
        simulation_params,
        material_properties,
    ):
        self.simulation_params = simulation_params
        self.material_properties = material_properties
        self.lambda_ = simulation_params.lambda_
        self.wX = simulation_params.wX
        self.wY = simulation_params.wY


def build_full_instance():
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialProperties()
    transducer = L125Transducer(simulation_params, material_properties)
    return transducer


def build_init_modified_instance():
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialProperties()
    transducer = L125TransducerMod(simulation_params, material_properties)
    return transducer


def test_full_instance():
    transducer = build_full_instance()
    assert isinstance(transducer, Transducer)
    assert isinstance(transducer, L125Transducer)


def test_properties():
    transducer = build_full_instance()
    isinstance(transducer.simulation_params, FocusedSimulationParams)
    isinstance(transducer.material_properties, MaterialProperties)


def test_make_transducer_surface_map():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    transducer = build_init_modified_instance()
    transducer._define_grid_params_based_on_ppw()

    transmitter_map, receiver_map = transducer._make_transducer_surface_map(
        nX=transducer.num_x,
        nY=transducer.num_y,
    )

    isinstance(transmitter_map, LinearTransmitterMap)
    isinstance(receiver_map, LinearReceiverMap)

    in_map = transmitter_map.in_map
    out_map = receiver_map.out_map

    test_utils.check_variable(
        mat_file_path=test_data_dir / "l125_inmap.mat",
        var_name="inmap",
        test_value=in_map,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "l125_outmap.mat",
        var_name="outmap",
        test_value=out_map,
    )

    incoords = geometry_utils.map_to_coordinates_matlab(in_map)
    outcoords = geometry_utils.map_to_coordinates_matlab(out_map)
    test_utils.check_variable(
        mat_file_path=test_data_dir / "l125_incoords_unsorted.mat",
        var_name="incoords",
        test_value=incoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "l125_outcoords_unsorted.mat",
        var_name="outcoords",
        test_value=outcoords,
    )

    incoords = geometry_utils.map_to_coords_with_sort(in_map)
    outcoords = geometry_utils.map_to_coords_with_sort(out_map)
    test_utils.check_variable(
        mat_file_path=test_data_dir / "l125_incoords_sorted.mat",
        var_name="incoords_sorted",
        test_value=incoords,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "l125_outcoords_sorted.mat",
        var_name="outcoords_sorted",
        test_value=outcoords,
    )
