import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.domains import geometry_utils
from fullwave_simulation.transducers import Transducer
from fullwave_simulation.transducers.linear_receiver_map import LinearReceiverMap
from fullwave_simulation.transducers.linear_transmitter_map import LinearTransmitterMap
from fullwave_simulation.utils import utils


class L125Transducer(Transducer):
    def __init__(
        self,
        simulation_params: Constant,
        material_properties: Constant,
    ):
        super().__init__(
            simulation_params=simulation_params,
            material_properties=material_properties,
        )
        self.lambda_ = simulation_params.lambda_
        self.wX = simulation_params.wX
        self.wY = simulation_params.wY
        self._define_grid_params_based_on_ppw()
        self._setup()

    def _define_grid_params_based_on_ppw(self):
        self.ppw = self.simulation_params.ppw

        self.num_x = np.round(self.wX / self.lambda_ * self.ppw).astype(int)
        self.num_y = np.round(self.wY / self.lambda_ * self.ppw).astype(int)

        self.dX = self.lambda_ / self.ppw
        self.dY = self.lambda_ / self.ppw
        self.beam_spacing = utils.matlab_round(self.simulation_params.spacing_m / self.dX)

    def _setup(self):
        self.transmitter_map, self.receiver_map = self._make_transducer_surface_map(
            self.num_x, self.num_y
        )
        self.inmap = self.transmitter_map.in_map
        self.outmap = self.receiver_map.out_map

        self.incoords = geometry_utils.map_to_coordinates_matlab(self.inmap)
        self.outcoords = geometry_utils.map_to_coordinates_matlab(self.outmap)

    def _make_transducer_surface_map(self, nX, nY):
        transmitter_map = LinearTransmitterMap(
            num_x=nX,
            num_y=nY,
            ppw=self.ppw,
            material_properties=self.material_properties,
            simulation_params=self.simulation_params,
        )
        receiver_map = LinearReceiverMap(
            num_x=nX,
            num_y=nY,
            beam_spacing=self.beam_spacing,
            in_map=transmitter_map.in_map,
            ppw=self.ppw,
            material_properties=self.material_properties,
            simulation_params=self.simulation_params,
        )
        return transmitter_map, receiver_map
