import numpy as np


class WholeMapReceiver:
    def __init__(
        self,
        num_x,
        num_y,
    ):
        self.num_x = num_x
        self.num_y = num_y
        self.out_map = self._calculate_outmap()

    def _calculate_outmap(self) -> np.ndarray:
        out_map = np.ones((self.num_x, self.num_y))
        return out_map
