from fullwave_simulation.constants import FocusedSimulationParams, MaterialProperties
from fullwave_simulation.transducers import L125Transducer, LinearTxWaveTransmitter
from fullwave_simulation.utils import test_utils


class WaveTransmitterMod(LinearTxWaveTransmitter):
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
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialProperties()

    # c52v_transducer = C52VTransducer(simulation_params, material_properties)
    l125_transducer = L125Transducer(simulation_params, material_properties)

    wave_transmitter = LinearTxWaveTransmitter(
        transducer=l125_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )
    return wave_transmitter


def build_init_modified_instance():
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialProperties()

    l125_transducer = L125Transducer(simulation_params, material_properties)

    wave_transmitter = WaveTransmitterMod(
        transducer=l125_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )
    return wave_transmitter


def test_instance():
    wave_transmitter = build_full_instance()
    assert isinstance(wave_transmitter, LinearTxWaveTransmitter)


def test_generate_t():
    wave_transmitter = build_full_instance()
    test_data_dir = test_utils.get_test_data_dir("transducers")
    for i in range(3):
        t_test = wave_transmitter.generate_t(i)
        mat_file_path = test_data_dir / f"linear_tx_t_t={i+1}.mat"
        var_name = "t"
        test_utils.check_variable(
            mat_file_path=mat_file_path, var_name=var_name, test_value=t_test
        )


def test_generate_transmit_pulse():
    wave_transmitter = build_full_instance()
    test_data_dir = test_utils.get_test_data_dir("transducers")
    for i in range(3):
        icvec_test = wave_transmitter.generate_transmit_pulse(i)
        mat_file_path = test_data_dir / f"linear_tx_icvec_t={i+1}.mat"
        var_name = "icvec"
        test_utils.check_variable(
            mat_file_path=mat_file_path, var_name=var_name, test_value=icvec_test
        )
