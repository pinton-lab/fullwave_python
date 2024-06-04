import numpy as np

from fullwave_simulation.constants import Constant


class Domain:
    def __init__(
        self,
        num_x: int,
        num_y: int,
        material_properties: Constant,
        simulation_params: Constant,
        name: str,
    ):
        self.num_x = num_x
        self.num_y = num_y
        self.material_properties = material_properties
        self.simulation_params = simulation_params
        self.name = name

        self.geometry = self._setup_geometry()

        self.rho_map = np.zeros((self.num_x, self.num_y))
        self.beta_map = np.zeros((self.num_x, self.num_y))
        self.c_map = np.zeros((self.num_x, self.num_y))
        self.a_map = np.zeros((self.num_x, self.num_y))

    def __call__(self):
        return {
            "rho_map": self.rho_map,
            "beta_map": self.beta_map,
            "c_map": self.c_map,
            "a_map": self.a_map,
        }

    def _setup_geometry(self):
        return np.zeros((self.num_x, self.num_y))

    # --- aliases ---

    @property
    def density_map(self):
        return self.rho_map

    @property
    def nonlinearity_map(self):
        return self.beta_map

    @property
    def speed_of_sound_map(self):
        return self.c_map

    @property
    def attenuation_map(self):
        return self.a_map
