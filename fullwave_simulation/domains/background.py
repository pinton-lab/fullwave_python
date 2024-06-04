import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.domains.domain import Domain


class Background(Domain):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        material_properties: Constant,
        simulation_params: Constant,
        name="background",
        background_domain_properties=None,
    ):
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )

        if background_domain_properties is not None:
            self.rho_map = (
                np.ones((self.num_x, self.num_y))
                * getattr(self.material_properties, background_domain_properties)["rho0"]
            )
            self.beta_map = (
                np.ones((self.num_x, self.num_y))
                * getattr(self.material_properties, background_domain_properties)["beta"]
            )
            self.c_map = (
                np.ones((self.num_x, self.num_y))
                * getattr(self.material_properties, background_domain_properties)["c0"]
            )
            self.a_map = (
                np.ones((self.num_x, self.num_y))
                * getattr(self.material_properties, background_domain_properties)["alpha"]
            )
        else:
            self.rho_map = np.ones((self.num_x, self.num_y)) * self.material_properties.rho0
            self.beta_map = np.ones((self.num_x, self.num_y)) * self.material_properties.beta0
            self.c_map = np.ones((self.num_x, self.num_y)) * self.material_properties.c0
            self.a_map = np.ones((self.num_x, self.num_y)) * self.material_properties.a0
        self.air_map = np.zeros((self.num_x, self.num_y))

    def _setup_geometry(self):
        geometry = np.ones((self.num_x, self.num_y))
        return geometry
