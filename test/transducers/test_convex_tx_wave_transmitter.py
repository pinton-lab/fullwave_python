from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.transducers import C52VTransducer, ConvexTxWaveTransmitter
from fullwave_simulation.utils import test_utils


class ConvexTxWaveTransmitterMod(ConvexTxWaveTransmitter):
    def __init__(
        self,
        transducer,
        simulation_params,
        material_properties,
        is_fsa=False,
    ):
        self.simulation_params = simulation_params
        self.material_properties = material_properties
        self.transducer = transducer
        self.is_fsa = is_fsa


def build_full_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)

    wave_transmitter = ConvexTxWaveTransmitter(
        c52v_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )
    return wave_transmitter


def build_init_modified_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)

    wave_transmitter = ConvexTxWaveTransmitterMod(
        c52v_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )
    return wave_transmitter


def test_instance():
    wave_transmitter = build_full_instance()
    assert isinstance(wave_transmitter, ConvexTxWaveTransmitter)


def test_determine_grid_params():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    wave_transmitter = build_init_modified_instance()
    wave_transmitter._determine_grid_params()
    nT = wave_transmitter.nT
    nT2 = wave_transmitter.nT2
    dT = wave_transmitter.dT
    fs = wave_transmitter.fs

    for test_value, var_name in [
        [nT, "nT"],
        [nT2, "nT2"],
        [dT, "dT"],
        [fs, "fs"],
    ]:
        pass

    test_utils.check_variable(
        mat_file_path=test_data_dir / f"{var_name}.mat",
        var_name=var_name,
        test_value=test_value,
    )


def test_focus_delays():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    wave_transmitter = build_init_modified_instance()
    focs = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "focs.mat",
        var_name="focs",
    )
    coords = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "incoords_delay.mat",
        var_name="incoords_delay",
    )

    cfl = 0.45
    varargin = 0
    delays = wave_transmitter._focus_delays(focs, coords, cfl, varargin)
    test_utils.check_variable(
        mat_file_path=test_data_dir / "focus_delays_dd.mat",
        var_name="dd",
        test_value=delays,
    )


def test_define_params_for_focused_sequecne():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    wave_transmitter = build_init_modified_instance()
    wave_transmitter._determine_grid_params()
    wave_transmitter._define_params_for_focused_sequecne()

    thetas = wave_transmitter.thetas

    test_utils.check_variable(
        mat_file_path=test_data_dir / "focus_thetas.mat",
        var_name="thetas",
        test_value=thetas,
    )


def test_focused_transmit_sequence():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    wave_transmitter = build_init_modified_instance()
    wave_transmitter._determine_grid_params()
    wave_transmitter._define_params_for_focused_sequecne()

    transmitter_dd = wave_transmitter._focused_transmit_sequence()

    focs = wave_transmitter.focs
    foc_origs = wave_transmitter.foc_origs

    test_utils.check_variable(
        mat_file_path=test_data_dir / "focs.mat",
        var_name="focs",
        test_value=focs,
    )

    test_utils.check_variable(
        mat_file_path=test_data_dir / "foc_origs.mat",
        var_name="foc_origs",
        test_value=foc_origs,
    )

    test_utils.check_variable(
        mat_file_path=test_data_dir / "focus_delays_dd_result.mat",
        var_name="transmitter_dd",
        test_value=transmitter_dd,
    )


def test_generate_transmit_pulse():
    test_data_dir = test_utils.get_test_data_dir("transducers")
    wave_transmitter = build_full_instance()
    nTic, icvec = wave_transmitter.generate_transmit_pulse()

    test_utils.check_variable(
        mat_file_path=test_data_dir / "nTic.mat",
        var_name="nTic",
        test_value=nTic,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "icvec.mat",
        var_name="icvec",
        test_value=icvec,
    )
