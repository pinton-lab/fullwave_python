import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.domains.domain import Domain
from fullwave_simulation.domains.geometry_utils import make_circle_idx


class ConvexTransmitterMap(Domain):
    def __init__(
        self,
        cen,
        rad,
        num_x: int,
        num_y: int,
        ppw,
        material_properties: Constant,
        simulation_params: Constant,
        name="transmitter",
        rho0=None,
        beta0=None,
        c0=None,
        a0=None,
    ):
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )
        self.cen = cen
        self.rad = rad
        self.ppw = ppw
        self.rho0 = rho0
        self.beta0 = beta0
        self.c0 = c0
        self.a0 = a0
        self.in_map = self._calculate_inmap()
        self.surface_label = self._make_transducer_surface_label(
            num_x=self.num_x,
            inmap=self.in_map,
            ppw=self.ppw,
        )
        (
            self.geometry,
            self.rho_map,
            self.beta_map,
            self.c_map,
            self.a_map,
        ) = self._calc_property_maps()

    def _make_transducer_surface_label(self, num_x, inmap, ppw):
        # Make a vector that indicates where the transducer surface is.
        surface = np.zeros(num_x)
        for i in range(num_x):
            # find where the transducer surface is
            j = np.where(inmap[i, :] == 1)[0]
            if len(j) != 0:
                j = j[-1] + 1
            else:
                j = 1
            surface[i] = j + np.round(ppw / 2 + 1e-9)
        return surface

    def _calculate_inmap(self) -> np.ndarray:
        # Make a circle that defines the transducer surface
        in_map = np.zeros((self.num_x, self.num_y))
        in_map[make_circle_idx(in_map.shape, self.cen, self.rad)] = 1

        # make outcoords from iccoords
        # Grab the coords on edge of the circle - larger circle for outcoords
        for i in range(self.num_x):
            # find inmap coords
            j = np.where(in_map[i, :] == 0)[0]
            j = j[0]

            in_map[i, 0 : np.maximum((j + 1) - 8, 0)] = 0
        return in_map

    def _calc_property_maps(self):
        geometry = np.zeros((self.num_x, self.num_y))
        rho_map = np.zeros((self.num_x, self.num_y))
        beta_map = np.zeros((self.num_x, self.num_y))
        c_map = np.zeros((self.num_x, self.num_y))
        a_map = np.zeros((self.num_x, self.num_y))
        for i in range(self.num_x):
            if (
                self.rho0 is not None
                and self.beta0 is not None
                and self.c0 is not None
                and self.a0 is not None
            ):
                rho_map[i, 0 : self.surface_label[i].astype(int)] = self.rho0
                beta_map[i, 0 : self.surface_label[i].astype(int)] = self.beta0
                c_map[i, 0 : self.surface_label[i].astype(int)] = self.c0
                a_map[i, 0 : self.surface_label[i].astype(int)] = self.a0
                geometry[i, 0 : self.surface_label[i].astype(int)] = 1
            else:
                rho_map[i, 0 : self.surface_label[i].astype(int)] = self.material_properties.rho0
                beta_map[i, 0 : self.surface_label[i].astype(int)] = self.material_properties.beta0
                c_map[i, 0 : self.surface_label[i].astype(int)] = self.material_properties.c0
                a_map[i, 0 : self.surface_label[i].astype(int)] = self.material_properties.a0
                geometry[i, 0 : self.surface_label[i].astype(int)] = 1
        return geometry, rho_map, beta_map, c_map, a_map

    def __call__(self):
        return {
            "rho_map": self.rho_map,
            "beta_map": self.beta_map,
            "c_map": self.c_map,
            "a_map": self.a_map,
            "input_map": self.in_map,
        }
