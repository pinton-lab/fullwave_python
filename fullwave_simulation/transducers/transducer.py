from fullwave_simulation.constants import Constant


class Transducer:
    def __init__(self, simulation_params: Constant, material_properties: Constant):
        self.simulation_params = simulation_params
        self.material_properties = material_properties
