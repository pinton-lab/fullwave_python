from pathlib import Path

from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains import (
    Background,
    DomainOrganizer,
    Phantom,
    Scatterer,
    WaterGel,
)
from fullwave_simulation.transducers import C52VTransducer
from fullwave_simulation.utils import test_utils


def build_full_instance():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)

    domain = Scatterer(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        material_properties=material_properties,
        simulation_params=simulation_params,
        transducer=c52v_transducer,
    )
    return domain


def test_full_instance():
    domain = build_full_instance()
    assert isinstance(domain, Scatterer)


def test_property():
    test_data_dir = test_utils.get_test_data_dir("domains")
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)

    scatter = Scatterer(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        material_properties=material_properties,
        simulation_params=simulation_params,
        transducer=c52v_transducer,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "scatter_map.mat",
        var_name="scatter_map",
        test_value=scatter.rho_map / material_properties.rho0,
    )


def test_integration():
    test_data_dir = test_utils.get_test_data_dir("domains")
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)
    domain_organizer = DomainOrganizer(material_properties=material_properties)

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
    domain_list = [
        background,
        water_gel,
        phantom,
        c52v_transducer.convex_transmitter_map,
    ]
    domain_organizer.register_domains(domain_list)
    domain_organizer.construct_domain()
    constructed_domain_dict = domain_organizer.constructed_domain_dict
    rho_map = constructed_domain_dict["rho_map"]
    test_utils.check_variable(
        mat_file_path=test_data_dir / "constructed_rho_map_csr-0035_seed42.mat",
        var_name="rmap_reg",
        test_value=rho_map,
    )
