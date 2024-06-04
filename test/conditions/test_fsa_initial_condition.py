from pathlib import Path

from fullwave_simulation.conditions import FSAInitialCondition
from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.transducers import C52VTransducer, ConvexTxWaveTransmitter
from fullwave_simulation.utils import test_utils


def build_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)
    wave_transmitter = ConvexTxWaveTransmitter(
        c52v_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )

    initial_condition = FSAInitialCondition(
        transducer=c52v_transducer,
        wave_transmitter=wave_transmitter,
    )
    return initial_condition


def test_instance():
    initial_condition = build_instance()
    assert isinstance(initial_condition, FSAInitialCondition)


def test_focus_coords_dd():
    test_data_dir = test_utils.get_test_data_dir("conditions")
    initial_condition = build_instance()
    event_id = 0

    icvec = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "icvec.mat",
        var_name="icvec",
    )[0]

    delay = initial_condition.wave_transmitter.transmitter_dd[:, event_id]
    varargin = 0
    icmat = initial_condition._focus_coords_dd(delay, icvec, varargin)

    test_utils.check_variable(
        mat_file_path=test_data_dir / "focus_coords_dd_icmat.mat",
        var_name="icmat",
        test_value=icmat,
    )


def test_generate_icmat():
    event_id = 0

    test_data_dir = test_utils.get_test_data_dir("conditions")
    initial_condition = build_instance()
    icmat = initial_condition.generate_icmat(event_id=event_id)

    test_utils.check_variable(
        mat_file_path=test_data_dir / "icmat.mat",
        var_name="icmat",
        test_value=icmat,
    )
