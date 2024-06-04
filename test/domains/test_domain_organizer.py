from collections import OrderedDict
from pathlib import Path

import numpy as np

from fullwave_simulation.constants import (
    Constant,
    FocusedSimulationParams,
    FSASimulationParams,
    MaterialProperties,
)
from fullwave_simulation.domains import (
    AbdominalWall,
    Background,
    DomainOrganizer,
    Phantom,
    WaterGel,
)
from fullwave_simulation.transducers import C52VTransducer, L125Transducer
from fullwave_simulation.utils import test_utils


def build_instance():
    material_properties = MaterialProperties()
    domain_organizer = DomainOrganizer(material_properties=material_properties)
    return domain_organizer


def test_instance():
    domain_organizer = build_instance()
    assert isinstance(domain_organizer, DomainOrganizer)
    assert isinstance(domain_organizer.registered_domain_dict, OrderedDict)
    assert isinstance(domain_organizer.constructed_domain_dict, OrderedDict)

    assert len(domain_organizer.registered_domain_dict) == 0
    assert len(domain_organizer.constructed_domain_dict) == 0


def test_register_maps_fsa():
    domain_organizer = build_instance()
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)

    background = Background(
        c52v_transducer.num_x,
        c52v_transducer.num_y,
        material_properties,
        simulation_params,
    )
    water_gel = WaterGel(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        depth=0.0124,
        dY=c52v_transducer.dY,
        material_properties=material_properties,
        simulation_params=simulation_params,
    )
    phantom = Phantom(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        material_properties=material_properties,
        simulation_params=simulation_params,
        dX=c52v_transducer.dX,
        dY=c52v_transducer.dY,
    )
    domain_list = [
        background,
        water_gel,
        phantom,
        c52v_transducer.convex_transmitter_map,
    ]
    domain_organizer.register_domains(domain_list)
    assert len(domain_organizer.registered_domain_dict) == 4
    assert len(domain_organizer.constructed_domain_dict) == 0
    for domain in domain_list:
        name = domain.name
        np.allclose(domain_organizer.registered_domain_dict[name]["rho_map"], domain.rho_map)
        np.allclose(domain_organizer.registered_domain_dict[name]["beta_map"], domain.beta_map)
        np.allclose(domain_organizer.registered_domain_dict[name]["c_map"], domain.c_map)
        np.allclose(domain_organizer.registered_domain_dict[name]["a_map"], domain.a_map)
        np.allclose(domain_organizer.registered_domain_dict[name]["geometry"], domain.geometry)


def test_construct_maps_fsa():
    test_dir = test_utils.get_test_data_dir("domains")
    domain_organizer = build_instance()
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    c52v_transducer = C52VTransducer(simulation_params, material_properties)

    background = Background(
        c52v_transducer.num_x,
        c52v_transducer.num_y,
        material_properties,
        simulation_params,
    )
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
    assert len(domain_organizer.registered_domain_dict) == 4
    assert len(domain_organizer.constructed_domain_dict) == 0
    domain_organizer.construct_domain()
    assert len(domain_organizer.registered_domain_dict) == 4
    assert len(domain_organizer.constructed_domain_dict) == 6

    constructed_domain_dict = domain_organizer.constructed_domain_dict
    for map_name, var_name in [
        ["rho_map", "rmap_reg"],
        ["beta_map", "beta_map"],
        ["c_map", "c_map"],
        ["a_map", "a_map"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_dir / f"constructed_{map_name}.mat",
            var_name=var_name,
            test_value=constructed_domain_dict[map_name],
        )


class MaterialPropertiesFocused(Constant):
    fat = {"bovera": 9.6, "alpha": 0.48, "ppower": 1.1, "c0": 1478, "rho0": 950}
    fat["beta"] = 1 + fat["bovera"] / 2

    # liver = {"bovera": 7.6, "alpha": 0.5, "ppower": 1.1, "c0": 1570, "rho0": 1064}
    # liver["beta"] = 1 + liver["bovera"] / 2

    muscle = {"bovera": 9, "alpha": 1.09, "ppower": 1.0, "c0": 1547, "rho0": 1050}
    muscle["beta"] = 1 + muscle["bovera"] / 2

    water = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1480, "rho0": 1000}
    water["beta"] = 1 + water["bovera"] / 2

    skin = {"bovera": 8, "alpha": 2.1, "ppower": 1, "c0": 1498, "rho0": 1000}
    skin["beta"] = 1 + skin["bovera"] / 2

    tissue = {"bovera": 9, "alpha": 0.5, "ppower": 1, "c0": 1540, "rho0": 1000}
    tissue["beta"] = 1 + tissue["bovera"] / 2

    connective = {"bovera": 8, "alpha": 1.57, "ppower": 1, "c0": 1613, "rho0": 1120}
    connective["beta"] = 1 + connective["bovera"] / 2

    blood = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1520, "rho0": 1000}
    blood["beta"] = 1 + blood["bovera"] / 2

    lung_fluid = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1440, "rho0": 1000}
    lung_fluid["beta"] = 1 + lung_fluid["bovera"] / 2

    c0 = 1540
    rho0 = 1000
    a0 = 0.5
    beta0 = 0


def test_construct_maps_focused():
    test_data_dir = test_utils.get_test_data_dir("domains")

    simulation_params = FocusedSimulationParams()
    material_properties = MaterialPropertiesFocused()

    l125_transducer = L125Transducer(simulation_params, material_properties)

    abdominal_wall = AbdominalWall(
        num_x=l125_transducer.num_x,
        num_y=l125_transducer.num_y,
        crop_depth=0.8e-2,
        start_depth=0.0,
        dY=l125_transducer.dY,
        dX=l125_transducer.dX,
        transducer=l125_transducer,
        abdominal_wall_mat_path=Path(
            "fullwave_simulation/domains/data/abdominal_wall/i2365f_etfw1.mat"
        ),
        material_properties=material_properties,
        simulation_params=simulation_params,
        apply_tissue_deformation=False,
        apply_tissue_compression=True,
        use_smoothing=True,
        skip_i0=False,
        background_domain_properties="lung_fluid",
    )

    background = Background(
        abdominal_wall.geometry.shape[0],
        l125_transducer.num_y,
        material_properties,
        simulation_params,
        background_domain_properties="water",
    )

    domain_organizer = DomainOrganizer(
        material_properties=material_properties,
        background_domain_properties="lung_fluid",
    )
    domain_organizer.register_domains(
        [
            background,
            abdominal_wall,
        ],
    )
    domain_organizer.construct_domain()

    constructed_domain_dict = domain_organizer.constructed_domain_dict

    for map_name, var_name in [
        ["c_map", "cmap"],
        ["rho_map", "rhomap"],
        ["a_map", "Amap"],
        # ["beta_map", "betamap"],
    ]:
        test_utils.check_variable(
            mat_file_path=test_data_dir / f"constructed_domain_focused_{var_name}.mat",
            var_name=var_name,
            test_value=constructed_domain_dict[map_name],
            export_diff_image_when_false=True,
        )
