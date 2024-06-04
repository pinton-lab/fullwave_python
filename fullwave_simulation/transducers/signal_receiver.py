from fullwave_simulation.constants import Constant
from fullwave_simulation.transducers.transducer import Transducer


class SignalReceiver(Transducer):
    def __init__(
        self,
        transducer: Transducer,
        simulation_params: Constant,
        material_properties: Constant,
    ):
        super().__init__(
            simulation_params=simulation_params,
            material_properties=material_properties,
        )
        self.transducer = transducer
