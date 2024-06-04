from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.transducers import Transducer


def build_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()
    transducer = Transducer(
        simulation_params=simulation_params,
        material_properties=material_properties,
    )
    return transducer


def test_instance():
    transducer = build_instance()
    assert isinstance(transducer, Transducer)


def test_properties():
    transducer = build_instance()
    isinstance(transducer.simulation_params, FSASimulationParams)
    isinstance(transducer.material_properties, MaterialProperties)
