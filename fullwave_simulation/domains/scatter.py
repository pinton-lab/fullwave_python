import numpy as np

from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains.domain import Domain
from fullwave_simulation.utils.utils import seed_everything


class Scatterer(Domain):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        material_properties: FSASimulationParams,
        simulation_params: MaterialProperties,
        transducer,
        name="scatter",
        num_scatter=18,
        seed=42,
    ):
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )
        self.seed = seed
        # seed_everything(seed)
        self.num_scatter = num_scatter
        self.transducer = transducer

        self.scatter_map, self.scat_count, self.scat_percent = self._setup_scatter_map(
            num_x=num_x,
            num_y=num_y,
            lambda_=transducer.lambda_,
            wY=transducer.wY,
            ncycles=self.simulation_params.ncycles,
            dX=transducer.dX,
            dY=transducer.dY,
            num_scat=num_scatter,
        )

        self.rho_map = self.scatter_map * self.material_properties.rho0
        self.beta_map = self.scatter_map * 0
        self.c_map = self.scatter_map * 0
        self.a_map = self.scatter_map * 0

    def _setup_geometry(self):
        geometry = np.ones((self.num_x, self.num_y))
        return geometry

    def _setup_scatter_map(
        self,
        num_x,
        num_y,
        lambda_,
        wY,
        ncycles,
        dX,
        dY,
        num_scat,
    ):
        res_cell = self._rescell2ds(lambda_, num_y / 2 * dY, wY, ncycles, dX, dY)
        scat_density = num_scat / res_cell

        seed_everything(self.seed)
        scatter_map = np.random.rand(num_y, num_x).T
        scatter_map = scatter_map / scat_density
        scatter_map[scatter_map > 1] = 0.5
        scatter_map = scatter_map - 0.5

        scat_count = len(scatter_map != 0)
        scat_percent = 100 * scat_count / (scatter_map.shape[0] * scatter_map.shape[0])
        scatter_map = scatter_map * self.geometry
        return scatter_map, scat_count, scat_percent

    def _rescell2ds(
        self,
        lambda_,
        dy,
        ay,
        n_cycles,
        dY,
        dZ,
    ):
        res_y = lambda_ * dy / ay
        res_z = lambda_ * n_cycles / 2
        res_cell = res_y / dY * res_z / dZ
        return res_cell
