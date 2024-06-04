from pathlib import Path

import numpy as np


class GenoutReader:
    def load_data_and_reshape(self, genout, num_x, num_y, num_t):
        return self._reshape(
            genout,
            num_x=num_x,
            num_y=num_y,
            num_t=num_t,
        )

    def load_and_reshape_from_path(self, genout_data_path, num_x, num_y, num_t):
        genout = self.load(genout_data_path)
        genout = self._reshape(genout, (num_t, num_y, num_x))
        return genout

    def load(self, genout_data_path: Path):
        genout = self._load_dat_data(genout_data_path)
        return genout

    def _reshape(self, genout, num_x, num_y, num_t):
        return np.reshape(
            genout,
            (num_t, num_y, num_x),
        )

    def _load_dat_data(dat_file_path, dtype=np.float32):
        if not dat_file_path.exists():
            raise ValueError(f"dat_file_path {dat_file_path} does not exist")
        return np.fromfile(dat_file_path, dtype=dtype)
