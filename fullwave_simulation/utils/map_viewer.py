from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MapViewer:
    def __init__(self, save_dir="test"):
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def view_map(
        self,
        map_data,
        title="Map",
        cmap="viridis",
        save_name_base=None,
        show=True,
        dpi=500,
        # extent=[5, 55, -7.5, 7.5],
        extent=None,
        # aspect=1 / 1.1,
        aspect=1.1,
        xticks=np.arange(120, 0, step=-20),
        yticks=np.arange(55.1, 5, step=-5),
        title_fontsize=8,
        label_fontsize=8,
    ):
        plt.imshow(
            map_data,
            cmap=cmap,
            extent=extent,
            aspect=aspect,
            interpolation="nearest",
        )
        plt.colorbar()
        plt.title(title, fontsize=title_fontsize)

        plt.xlabel("width [mm]", fontsize=label_fontsize)
        plt.ylabel("Depth [mm]", fontsize=label_fontsize)
        # plt.xticks(xticks)
        # plt.yticks(yticks)
        plt.tight_layout()

        if save_name_base is not None:
            plt.savefig(self._save_dir / f"{save_name_base}.png", dpi=dpi)
        if show:
            plt.show()
        plt.cla()
        plt.clf()

    def view_bmode(self, map_data):
        self.view_map(map_data)
