import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.domains.domain import Domain
from fullwave_simulation.utils import utils


class LinearReceiverMap(Domain):
    def __init__(
        self,
        num_x,
        num_y,
        beam_spacing,
        in_map,
        ppw,
        material_properties: Constant,
        simulation_params: Constant,
        name="receiver",
    ):
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )
        self.in_map = in_map
        self.ppw = ppw
        self.beam_spacing = beam_spacing
        self.out_map = self._calculate_outmap()
        self.surface_label = self._make_transducer_surface_label(
            num_x=self.num_x,
            inmap=self.in_map,
            ppw=self.ppw,
        )
        (
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

    def _calculate_outmap(self) -> np.ndarray:
        out_map = np.zeros((self.num_x, self.num_y))
        out_map[
            utils.matlab_round(self.num_x / 2)
            - utils.matlab_round(self.simulation_params.txducer_aperture * self.beam_spacing / 2)
            - 1 : utils.matlab_round(self.num_x / 2)
            + utils.matlab_round(self.simulation_params.txducer_aperture * self.beam_spacing / 2)
            - 1,
            3,
        ] = 1
        return out_map

    def _calc_property_maps(self):
        rho_map = np.zeros((self.num_x, self.num_y))
        beta_map = np.zeros((self.num_x, self.num_y))
        c_map = np.zeros((self.num_x, self.num_y))
        a_map = np.zeros((self.num_x, self.num_y))
        for i in range(self.num_x):
            rho_map[i, 0 : self.surface_label[i].astype(int)] = self.material_properties.rho0
            beta_map[i, 0 : self.surface_label[i].astype(int)] = self.material_properties.beta0
            c_map[i, 0 : self.surface_label[i].astype(int)] = self.material_properties.c0
            a_map[i, 0 : self.surface_label[i].astype(int)] = self.material_properties.a0
        return rho_map, beta_map, c_map, a_map

    def __call__(self):
        return {
            "rho_map": self.rho_map,
            "beta_map": self.beta_map,
            "c_map": self.c_map,
            "a_map": self.a_map,
            "input_map": self.in_map,
        }
