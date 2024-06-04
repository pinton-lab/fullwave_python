import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.domains.domain import Domain
from fullwave_simulation.domains.geometry_utils import make_circle_idx
from fullwave_simulation.utils.utils import matlab_round


class Phantom(Domain):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        material_properties: Constant,
        simulation_params: Constant,
        dX: float,
        dY: float,
        depth_phantom_in_meter=0.095 + 0.0124,
        lat_phantom_in_meter=0.12,
        base_circle_depth_in_meter=0.03,
        name="phantom",
        background_domain_properties=None,
    ):
        self.dX = dX
        self.dY = dY
        self.depth_phantom_in_meter = depth_phantom_in_meter
        self.lat_phantom_in_meter = lat_phantom_in_meter
        self.base_circle_depth_in_meter = base_circle_depth_in_meter
        super().__init__(
            num_x=num_x,
            num_y=num_y,
            material_properties=material_properties,
            simulation_params=simulation_params,
            name=name,
        )
        self.background_domain_properties = background_domain_properties

        if self.background_domain_properties is not None:
            self.rho_map = (
                np.ones((self.num_x, self.num_y))
                * getattr(self.material_properties, self.background_domain_properties)["rho0"]
            )
            self.c_map = (
                np.ones((self.num_x, self.num_y))
                * getattr(self.material_properties, self.background_domain_properties)["c0"]
            )
            self.a_map = (
                np.ones((self.num_x, self.num_y))
                * getattr(self.material_properties, self.background_domain_properties)["alpha"]
            )
        else:
            self.rho_map = (self.geometry > 0) * self.material_properties.rho0
            self.c_map = (self.geometry > 0) * self.material_properties.c0
            self.a_map = (self.geometry > 0) * self.material_properties.a0

    def _setup_geometry(
        self,
    ):
        view = 0
        fp_lat = matlab_round(self.lat_phantom_in_meter / self.dX)
        fp_dep = matlab_round((self.depth_phantom_in_meter) / self.dY)
        fp_maps = np.zeros((fp_lat, fp_dep))
        radius = np.array(
            [
                matlab_round(0.016668 / self.dX / 2),
                matlab_round(0.010404 / self.dX / 2),
                matlab_round(0.00649 / self.dX / 2),
                matlab_round(0.00405 / self.dX / 2),
                matlab_round(0.00253 / self.dX / 2),
                matlab_round(0.00158 / self.dX / 2),
            ]
        )
        center1 = np.array(
            [
                matlab_round(0.015 / self.dX) - 1,
                matlab_round((self.base_circle_depth_in_meter + 0.0124) / self.dY),
            ]
        )
        center2 = np.array(
            [
                matlab_round(0.045 / self.dX) - 1,
                matlab_round((self.base_circle_depth_in_meter + 0.0124) / self.dY),
            ]
        )
        center3 = np.array(
            [
                matlab_round(0.075 / self.dX) - 1,
                matlab_round((self.base_circle_depth_in_meter + 0.0124) / self.dY),
            ]
        )
        center4 = np.array(
            [
                matlab_round(0.105 / self.dX) - 1,
                matlab_round((self.base_circle_depth_in_meter + 0.0124) / self.dY),
            ]
        )
        center5 = np.array(
            [
                matlab_round(0.015 / self.dX) - 1,
                matlab_round((2 * (self.base_circle_depth_in_meter + 0.0124)) / self.dY),
            ]
        )
        center6 = np.array(
            [
                matlab_round(0.045 / self.dX) - 1,
                matlab_round((2 * (self.base_circle_depth_in_meter + 0.0124)) / self.dY),
            ]
        )
        center7 = np.array(
            [
                matlab_round(0.075 / self.dX) - 1,
                matlab_round((2 * (self.base_circle_depth_in_meter + 0.0124)) / self.dY),
            ]
        )
        center8 = np.array(
            [
                matlab_round(0.105 / self.dX) - 1,
                matlab_round((2 * (self.base_circle_depth_in_meter + 0.0124)) / self.dY),
            ]
        )
        id1 = np.logical_or(
            make_circle_idx(np.array([fp_lat, fp_dep]), center1, radius[view]),
            make_circle_idx(np.array([fp_lat, fp_dep]), center5, radius[-1 - view]),
        )
        id2 = np.logical_or(
            make_circle_idx(np.array([fp_lat, fp_dep]), center2, radius[-1 - view]),
            make_circle_idx(np.array([fp_lat, fp_dep]), center6, radius[view]),
        )
        id3 = np.logical_or(
            make_circle_idx(np.array([fp_lat, fp_dep]), center3, radius[view]),
            make_circle_idx(np.array([fp_lat, fp_dep]), center7, radius[-1 - view]),
        )
        id4 = np.logical_or(
            make_circle_idx(np.array([fp_lat, fp_dep]), center4, radius[-1 - view]),
            make_circle_idx(np.array([fp_lat, fp_dep]), center8, radius[view]),
        )
        fp_maps[id1] = 1
        fp_maps[id2] = 2
        fp_maps[id3] = 3
        fp_maps[id4] = 4
        # if self.num_x is bigger than the fp_maps, then we need to pad the fp_maps
        if self.num_x > fp_maps.shape[0]:
            fp_maps = np.pad(
                fp_maps,
                ((0, self.num_x - fp_maps.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        # if self.num_y is bigger than the fp_maps, then we need to pad the fp_maps
        if self.num_y > fp_maps.shape[1]:
            fp_maps = np.pad(
                fp_maps,
                ((0, 0), (0, self.num_y - fp_maps.shape[1])),
                mode="constant",
                constant_values=0,
            )

        geometry = fp_maps[: self.num_x, : self.num_y]
        return geometry


class PhatomLateral(Phantom):
    def _setup_geometry(
        self,
    ):
        view = 0
        fp_lat = matlab_round(self.lat_phantom_in_meter / self.dX)
        fp_dep = matlab_round((self.depth_phantom_in_meter) / self.dY)
        fp_maps = np.zeros((fp_lat, fp_dep))
        radius = np.array(
            [
                matlab_round(1e-2 / self.dX / 2),
            ]
        )
        center1 = np.array(
            [
                matlab_round((fp_lat / 2)),
                matlab_round((self.base_circle_depth_in_meter) / self.dY),
            ]
        )
        center2 = np.array(
            [
                matlab_round((fp_lat / 2)),
                matlab_round((self.base_circle_depth_in_meter + 0.02) / self.dY),
            ]
        )

        center3 = np.array(
            [
                matlab_round((fp_lat / 2)),
                matlab_round((self.base_circle_depth_in_meter + 0.04) / self.dY),
            ]
        )

        id1 = make_circle_idx(np.array([fp_lat, fp_dep]), center1, radius[view])
        id2 = make_circle_idx(np.array([fp_lat, fp_dep]), center2, radius[view])
        id3 = make_circle_idx(np.array([fp_lat, fp_dep]), center3, radius[view])

        fp_maps[id1] = 1
        fp_maps[id2] = 2
        fp_maps[id3] = 3
        # if self.num_x is bigger than the fp_maps, then we need to pad the fp_maps
        if self.num_x > fp_maps.shape[0]:
            fp_maps = np.pad(
                fp_maps,
                ((0, self.num_x - fp_maps.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        # if self.num_y is bigger than the fp_maps, then we need to pad the fp_maps
        if self.num_y > fp_maps.shape[1]:
            fp_maps = np.pad(
                fp_maps,
                ((0, 0), (0, self.num_y - fp_maps.shape[1])),
                mode="constant",
                constant_values=0,
            )

        geometry = fp_maps[: self.num_x, : self.num_y]
        return geometry


def main():
    from fullwave_simulation.constants import FSASimulationParams, MaterialProperties

    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    freq_div = 1
    c0 = 1540
    f0 = 3700000.0 / freq_div
    lambda_ = c0 / f0
    wX = 0.12
    wY = 0.04 + 0.0124

    ptch = 0.000508
    subelem_ptch = 15 / freq_div

    ppw = lambda_ / (ptch / subelem_ptch)

    num_x = np.round(wX / lambda_ * ppw).astype(int)
    num_y = np.round(wY / lambda_ * ppw).astype(int)
    dX = lambda_ / ppw

    phantom = Phantom(
        num_x,
        num_y,
        material_properties,
        simulation_params,
        dX,
    )
    print(phantom())


if __name__ == "__main__":
    main()
