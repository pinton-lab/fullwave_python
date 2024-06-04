import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.domains.domain import Domain


class LinearTransmitterMap(Domain):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        ppw,
        material_properties: Constant,
        simulation_params: Constant,
        name="transmitter",
    ):
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )
        self.ppw = ppw
        self.in_map = self._calculate_inmap()

    def _calculate_inmap(self) -> np.ndarray:
        in_map = np.zeros((self.num_x, self.num_y))
        in_map[:, 0:3] = 1
        return in_map
