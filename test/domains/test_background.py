from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains import Background


def build_full_instance():
    num_x = 100
    num_y = 100
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()
    domain = Background(
        num_x=num_x,
        num_y=num_y,
        simulation_params=simulation_params,
        material_properties=material_properties,
    )
    return domain


def test_full_instance():
    domain = build_full_instance()
    assert isinstance(domain, Background)


def test_properties():
    domain = build_full_instance()
    assert domain.num_x == 100
    assert domain.num_y == 100
    assert domain.name == "background"
    assert isinstance(domain.simulation_params, FSASimulationParams)
    assert isinstance(domain.material_properties, MaterialProperties)

    assert domain.geometry.shape == (100, 100)
    assert domain.rho_map.shape == (100, 100)
    assert domain.beta_map.shape == (100, 100)
    assert domain.c_map.shape == (100, 100)
    assert domain.a_map.shape == (100, 100)

    assert domain.density_map.shape == (100, 100)
    assert domain.nonlinearity_map.shape == (100, 100)
    assert domain.speed_of_sound_map.shape == (100, 100)
    assert domain.attenuation_map.shape == (100, 100)

    assert isinstance(domain(), dict)
