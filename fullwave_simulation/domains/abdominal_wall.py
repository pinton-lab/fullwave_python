from pathlib import Path

import cv2
import numpy as np
import scipy

from fullwave_simulation.constants import FSASimulationParams, MaterialProperties
from fullwave_simulation.domains.domain import Domain
from fullwave_simulation.utils import utils


class AbdominalWall(Domain):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        crop_depth: float,
        start_depth: float,
        dX: float,
        dY: float,
        transducer,
        abdominal_wall_mat_path: Path,
        material_properties: MaterialProperties,
        simulation_params: FSASimulationParams,
        apply_tissue_deformation: bool = False,
        apply_tissue_compression: bool = True,
        name="abdominal_wall",
        background_domain_properties="water",
        use_smoothing=False,
        skip_i0=True,
        sequence_type="focused",
        use_center_region=False,
        ppw=12,
    ):
        self.crop_depth = crop_depth
        self.start_depth = start_depth
        self.dX = dX
        self.dY = dY
        self.transducer = transducer
        self.abdominal_wall_mat_path = abdominal_wall_mat_path
        self.apply_tissue_deformation = apply_tissue_deformation
        self.ppw = ppw
        if apply_tissue_deformation:
            self.tranducer_surface = self.transducer.convex_transmitter_map.surface_label.astype(
                int
            )
        self.apply_tissue_compression = apply_tissue_compression
        self.use_smoothing = use_smoothing
        self.skip_i0 = skip_i0
        self.sequence_type = sequence_type
        self.use_center_region = use_center_region
        self.background_domain_properties = background_domain_properties
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )
        self.rho_map, self.c_map, self.a_map, self.beta_map = self._setup_maps()

    def _setup_geometry(self):
        mat_data = scipy.io.loadmat(self.abdominal_wall_mat_path)
        abdominal_wall_properties = mat_data["cut"].T.astype(float)
        # this is the pixel size in interpd Visual Human slice.
        dm = 0.33e-3 / 4

        if self.apply_tissue_compression:
            compression_ratio = 0.655
        else:
            compression_ratio = 1
        interpolation_x = dm / self.dX
        interpolation_y = compression_ratio * dm / self.dY

        x_new = utils.matlab_round(abdominal_wall_properties.shape[0] * interpolation_x)

        abdominal_wall_properties = utils.matlab_interp2easy(
            abdominal_wall_properties,
            interpolation_x=interpolation_x,
            interpolation_y=interpolation_y,
        )

        crop_depth_index = utils.matlab_round(self.crop_depth / self.dY) - 1
        start_depth_index = utils.matlab_round(self.start_depth / self.dY)

        abdominal_wall_properties = abdominal_wall_properties[
            :, crop_depth_index : crop_depth_index + self.num_y
        ]

        if self.sequence_type == "fsa" or self.sequence_type == "FSA":
            domain_width = self.num_x
            if self.use_center_region:
                center_x = abdominal_wall_properties.shape[0] // 2
                abdominal_wall_properties = abdominal_wall_properties[
                    center_x - domain_width // 2 : center_x + domain_width // 2, : self.num_y
                ]
            else:
                abdominal_wall_properties = abdominal_wall_properties[:domain_width, : self.num_y]
        elif self.sequence_type == "focused":
            domain_width = x_new
            abdominal_wall_properties = abdominal_wall_properties[:, : self.num_y]
        elif self.sequence_type == "plane":
            domain_width = self.num_x
            if self.use_center_region:
                center_x = abdominal_wall_properties.shape[0] // 2
                abdominal_wall_properties = abdominal_wall_properties[
                    center_x - domain_width // 2 : center_x + domain_width // 2, : self.num_y
                ]
        else:
            raise ValueError("Invalid sequence type")

        geometry = np.zeros((domain_width, self.num_y))

        if self.apply_tissue_deformation:
            for i in range(domain_width):
                abdominal_wall_properties[
                    i, self.tranducer_surface[i] - 3 :
                ] = abdominal_wall_properties[i, 0 : -self.tranducer_surface[i] + 3]

                abdominal_wall_properties[i, 0 : self.tranducer_surface[i] - 3] = 0

        geometry[
            : abdominal_wall_properties.shape[0],
            start_depth_index : start_depth_index + abdominal_wall_properties.shape[1],
        ] = abdominal_wall_properties
        return geometry

    def _setup_maps(self):
        rho_map = np.ones_like(self.geometry) * self.material_properties.water["rho0"]
        c_map = np.ones_like(self.geometry) * self.material_properties.water["c0"]
        a_map = np.ones_like(self.geometry) * self.material_properties.water["alpha"]
        beta_map = np.ones_like(self.geometry) * self.material_properties.water["beta"]

        for i, tissue_name in enumerate(
            [
                self.background_domain_properties,
                "connective",
                "muscle",
                "fat",
                "connective",
                "connective",
            ],
        ):
            if i == 0 and self.skip_i0:
                continue
            target_index = np.where(self.geometry == i)
            rho_map[target_index] = getattr(self.material_properties, tissue_name)["rho0"]
            c_map[target_index] = getattr(self.material_properties, tissue_name)["c0"]
            a_map[target_index] = getattr(self.material_properties, tissue_name)["alpha"]
        if self.use_smoothing:
            # use gaussian smoothing to smooth the abdominal wall
            sigma = (5 / 10) ** 2 * self.ppw / 2
            rho_map_blurred = utils.matlab_gaussian_filter(rho_map, sigma=sigma)
            c_map_blurred = utils.matlab_gaussian_filter(c_map, sigma=sigma)
            a_map_blurred = utils.matlab_gaussian_filter(a_map, sigma=sigma)
            # update the geometry following the gaussian filter result
            # c_map_blurred_boundary0 = np.logical_xor(
            #     (self.geometry > 0),
            #     (c_map_blurred > (self.material_properties.lung_fluid["c0"] + 10)),
            # )

            c_map_blurred_boundary = np.logical_and(
                ~np.isclose(c_map_blurred, self.material_properties.lung_fluid["c0"], rtol=0.007),
                (self.geometry == 0),
            )
            rho_map_blurred_boundary = np.logical_and(
                ~np.isclose(
                    rho_map_blurred, self.material_properties.lung_fluid["rho0"], rtol=0.007
                ),
                (self.geometry == 0),
            )
            a_map_blurred_boundary = np.logical_and(
                ~np.isclose(
                    a_map_blurred, self.material_properties.lung_fluid["alpha"], rtol=0.007
                ),
                (self.geometry == 0),
            )
            blurred_boundary = np.logical_or(
                c_map_blurred_boundary,
                np.logical_or(rho_map_blurred_boundary, a_map_blurred_boundary),
            )

            # self.geometry[blurred_boundary] = 1
            # self.geometry[c_map_blurred_boundary] = 1

            rho_map = rho_map_blurred
            c_map = c_map_blurred
            a_map = a_map_blurred
        return rho_map, c_map, a_map, beta_map
