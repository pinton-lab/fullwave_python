import numpy as np

from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains.domain import Domain
from fullwave_simulation.utils.utils import matlab_round


class WaterGel(Domain):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        depth: float,
        dY: float,
        material_properties: MaterialProperties,
        simulation_params: FSASimulationParams,
        name="water_gel",
    ):
        self.depth = depth
        self.dY = dY
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )
        self.rho_map = self.geometry * self.material_properties.water["rho0"]
        self.c_map = self.geometry * self.material_properties.water["c0"]
        self.a_map = self.geometry * self.material_properties.water["alpha"]
        self.beta_map = self.geometry * self.material_properties.beta0

    def _setup_geometry(self):
        geometry = np.zeros((self.num_x, self.num_y))
        geometry[:, 0 : matlab_round(self.depth / self.dY)] = 1
        return geometry
