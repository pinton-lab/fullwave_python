from collections import OrderedDict
from typing import List

import numpy as np

from fullwave_simulation.domains.domain import Domain


class DomainOrganizer:
    def __init__(
        self,
        material_properties,
        ignore_non_linearity=False,
        background_domain_properties="water",
    ):
        self.material_properties = material_properties
        self.registered_domain_dict = OrderedDict()
        self.constructed_domain_dict = OrderedDict()
        self.ignore_non_linearity = ignore_non_linearity
        self.background_domain_properties = background_domain_properties

    def register_domains(self, domain_list: List[Domain]):
        for domain in domain_list:
            domain_name = domain.name
            rho_map = domain.rho_map
            beta_map = domain.beta_map
            c_map = domain.c_map
            a_map = domain.a_map
            geometry = domain.geometry
            if hasattr(domain, "air_map"):
                air_map = domain.air_map
            else:
                air_map = np.zeros_like(geometry)

            self.registered_domain_dict[domain_name] = {
                "rho_map": rho_map,
                "beta_map": beta_map,
                "c_map": c_map,
                "a_map": a_map,
                "air_map": air_map,
                "geometry": geometry,
            }

    def construct_domain(self):
        # construct maps from registered maps. combine the maps like an image stecker.
        # The first map is the background map.
        # Replace with the number that came later but do not replace the later number with a zero.
        var_name_dict = {
            "rho_map": "rho0",
            "beta_map": "beta",
            "c_map": "c0",
            "a_map": "alpha",
        }
        for map_name in ["rho_map", "beta_map", "c_map", "a_map", "air_map"]:
            base_map_data = self.registered_domain_dict["background"][map_name].copy()
            for domain_name in self.registered_domain_dict.keys():
                if domain_name == "background":
                    continue
                elif domain_name == "abdominal_wall":
                    if map_name == "rho_map":
                        rtol = 1e-6
                    else:
                        rtol = 7e-3

                    if map_name == "air_map":
                        non_zero_index = np.where(
                            np.logical_or(
                                ~np.isclose(self.registered_domain_dict[domain_name][map_name], 0),
                                self.registered_domain_dict[domain_name]["geometry"] != 0,
                            )
                        )
                    else:
                        non_zero_index = np.where(
                            np.logical_or(
                                ~np.isclose(
                                    self.registered_domain_dict[domain_name][map_name],
                                    getattr(
                                        self.material_properties, self.background_domain_properties
                                    )[var_name_dict[map_name]],
                                    rtol=rtol,
                                ),
                                self.registered_domain_dict[domain_name]["geometry"] != 0,
                            )
                        )
                else:
                    non_zero_index = np.where(
                        self.registered_domain_dict[domain_name]["geometry"] != 0,
                    )
                base_map_data[non_zero_index] = self.registered_domain_dict[domain_name][map_name][
                    non_zero_index
                ]
            self.constructed_domain_dict[map_name] = base_map_data.copy()

        if self.ignore_non_linearity:
            self.constructed_domain_dict["beta_map"] = np.zeros_like(
                self.constructed_domain_dict["beta_map"],
            )

        base_geometry_data = np.zeros_like(self.registered_domain_dict["background"]["geometry"])
        for i, domain_name in enumerate(self.registered_domain_dict.keys()):
            geometry_non_zero_index = np.where(
                self.registered_domain_dict[domain_name]["geometry"] != 0,
            )
            base_geometry_data[geometry_non_zero_index] = i
        self.constructed_domain_dict["geometry"] = base_geometry_data.copy()
