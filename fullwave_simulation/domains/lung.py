from pathlib import Path

import numpy as np
import scipy

from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains.domain import Domain
from fullwave_simulation.utils import utils


class Lung(Domain):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        start_depth: float,
        dX: float,
        dY: float,
        transducer,
        lung_mat_path: Path,
        material_properties: MaterialProperties,
        simulation_params: FSASimulationParams,
        name="lung",
        background_domain_properties="water",
        use_smoothing=False,
        skip_i0=True,
    ):
        self.start_depth = start_depth
        self.dX = dX
        self.dY = dY
        self.transducer = transducer
        self.lung_mat_path = lung_mat_path
        self.use_smoothing = use_smoothing
        self.background_domain_properties = background_domain_properties
        self.skip_i0 = skip_i0
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )
        self.rho_map, self.c_map, self.a_map, self.beta_map, self.air_map = self._setup_maps()

    def _setup_geometry(self):
        mat_data = scipy.io.loadmat(self.lung_mat_path)
        lung_properties = mat_data["lung"].astype(float)
        lung_properties = np.asanyarray(lung_properties.T)
        # this is the pixel size in interpd Visual Human slice.
        lung_properties = lung_properties[:, 21:]
        lung_properties[lung_properties == 1] = 2
        lung_properties[lung_properties == 0] = 1

        dm = 2.02e-2 / 814
        visceral_pleura_thickness = 0.1e-3  # 0.1 mm

        interpolation_x = dm / self.dX
        interpolation_y = dm / self.dY

        lung_properties = utils.matlab_interp2easy(
            lung_properties,
            interpolation_x=interpolation_x,
            interpolation_y=interpolation_y,
        )
        zero_property_index = np.where(lung_properties == 1)
        zero_property_x = zero_property_index[0]
        zero_property_y = zero_property_index[1]
        tissue_boarder_y = np.array(
            [zero_property_y[zero_property_x == i].min() for i in range(lung_properties.shape[0])]
        )
        for i in range(lung_properties.shape[0]):
            lung_properties[i, : tissue_boarder_y[i]] = 3

            # assign connective tissue properties to the 0.2 mm border
            lung_properties[
                i,
                tissue_boarder_y[i]
                - utils.matlab_round(visceral_pleura_thickness / self.dY) : tissue_boarder_y[i],
            ] = 4

        lung_properties = lung_properties[:, : utils.matlab_round(529 * dm / self.dY)]

        domain_width = self.num_x
        lung_properties = lung_properties[:domain_width, : self.num_y]

        start_depth_index = utils.matlab_round(self.start_depth / self.dY)

        geometry = np.zeros((domain_width, self.num_y))

        lung_position_x_start = domain_width // 2 - lung_properties.shape[0] // 2
        lung_position_x_end = lung_position_x_start + lung_properties.shape[0]
        lung_position_y_start = start_depth_index
        lung_position_y_end = start_depth_index + lung_properties.shape[1]

        geometry[
            lung_position_x_start:lung_position_x_end, lung_position_y_start:lung_position_y_end
        ] = lung_properties

        geometry[:lung_position_x_start, lung_position_y_start + 1 + tissue_boarder_y[0] :] = 5
        geometry[lung_position_x_end:, lung_position_y_start + 1 + tissue_boarder_y[-1] :] = 5
        geometry[:, lung_position_y_end:] = 5

        geometry[
            :lung_position_x_start,
            lung_position_y_start : lung_position_y_start + 1 + tissue_boarder_y[0],
        ] = 3
        geometry[
            lung_position_x_end:,
            lung_position_y_start : lung_position_y_start + 1 + tissue_boarder_y[-1],
        ] = 3

        geometry[
            :lung_position_x_start,
            lung_position_y_start
            + 1
            + tissue_boarder_y[0]
            - utils.matlab_round(visceral_pleura_thickness / self.dY) : lung_position_y_start
            + 1
            + tissue_boarder_y[0],
        ] = 4

        geometry[
            lung_position_x_end:,
            lung_position_y_start
            + 1
            + tissue_boarder_y[-1]
            - utils.matlab_round(visceral_pleura_thickness / self.dY) : lung_position_y_start
            + 1
            + tissue_boarder_y[-1],
        ] = 4
        return geometry

    def _setup_maps(self):
        rho_map = np.ones_like(self.geometry) * self.material_properties.water["rho0"]
        c_map = np.ones_like(self.geometry) * self.material_properties.water["c0"]
        a_map = np.ones_like(self.geometry) * self.material_properties.water["alpha"]
        beta_map = np.ones_like(self.geometry) * self.material_properties.water["beta"]
        air_map = np.zeros_like(self.geometry)

        for i, tissue_name in enumerate(
            [
                self.background_domain_properties,
                "tissue",
                "lung_air",
                "lung_fluid",
                "connective",
                "lung_air",
            ],
        ):
            if i == 0 and self.skip_i0:
                continue
            target_index = np.where(self.geometry == i)
            rho_map[target_index] = getattr(self.material_properties, tissue_name)["rho0"]
            c_map[target_index] = getattr(self.material_properties, tissue_name)["c0"]
            a_map[target_index] = getattr(self.material_properties, tissue_name)["alpha"]

        air_map[self.geometry == 2] = 1
        air_map[self.geometry == 5] = 1

        if self.use_smoothing:
            # use gaussian smoothing to smooth the abdominal wall
            sigma = (5 / 10) ** 2 * self.simulation_params.ppw / 2
            rho_map = utils.matlab_gaussian_filter(rho_map, sigma=sigma)
            c_map = utils.matlab_gaussian_filter(c_map, sigma=sigma)
            a_map = utils.matlab_gaussian_filter(a_map, sigma=sigma)
            beta_map = utils.matlab_gaussian_filter(beta_map, sigma=sigma)

        return rho_map, c_map, a_map, beta_map, air_map
