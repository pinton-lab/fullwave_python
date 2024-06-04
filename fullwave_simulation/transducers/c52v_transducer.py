import numpy as np

from fullwave_simulation.constants import Constant
from fullwave_simulation.domains import geometry_utils
from fullwave_simulation.transducers import Transducer
from fullwave_simulation.transducers.covex_receiver_map import ConvexReceiverMap
from fullwave_simulation.transducers.covex_transmitter_map import ConvexTransmitterMap


class C52VTransducer(Transducer):
    def __init__(
        self,
        simulation_params: Constant,
        material_properties: Constant,
        rho0=1000,
        beta0=0,
        c0=1540,
        a0=0.5,
        element_pitch=0.000508,
    ):
        super().__init__(
            simulation_params=simulation_params,
            material_properties=material_properties,
        )
        self.lambda_ = simulation_params.lambda_
        self.freq_div = simulation_params.freq_div
        self.wX = simulation_params.wX
        self.wY = simulation_params.wY
        self.rho0 = rho0
        self.beta0 = beta0
        self.c0 = c0
        self.a0 = a0
        self.element_pitch = element_pitch
        self._define_grid_params_based_on_ppw()
        self._compute_transducer_variables()
        self._setup()

    def _define_grid_params_based_on_ppw(self):
        # define grid paramaters based on ppw
        self.subelem_ptch = 15 / self.freq_div

        self.ppw = self.lambda_ / (self.element_pitch / self.subelem_ptch)

        self.num_x = np.round(self.wX / self.lambda_ * self.ppw).astype(int)
        self.num_y = np.round(self.wY / self.lambda_ * self.ppw).astype(int)
        self.dX = self.lambda_ / self.ppw
        self.dY = self.lambda_ / self.ppw

    def _compute_transducer_variables(self):
        # define variables
        zero_offset = 0.0124
        self.npx = self.simulation_params.nevents
        self.rad = 0.04957 / self.dY
        # self.rad = 0.049 / self.dY
        self.dTheta = np.arctan2(self.subelem_ptch, self.rad)

        self.cen = np.array([self.num_x / 2, zero_offset / self.dY - self.rad])

    def _setup(self):
        self.thetas = self._define_theta_at_center_of_elements(
            dTheta=self.dTheta,
            npx=self.npx,
        )
        self.convex_transmitter_map, self.convex_receiver_map = self._make_transducer_surface_map(
            nX=self.num_x,
            nY=self.num_y,
            cen=self.cen,
            rad=self.rad,
        )
        self.inmap = self.convex_transmitter_map.in_map
        self.outmap = self.convex_receiver_map.out_map
        self.incoords = geometry_utils.map_to_coords_with_sort(self.inmap)
        self.outcoords = geometry_utils.map_to_coords_with_sort(self.outmap)
        (
            self.thetas_in,
            self.thetas_out,
            self.incoords,
            self.outcoords,
            self.incoords2,
            self.outcoords2,
            self.nOutPx,
            self.nInPx,
        ) = self._assign_transducer_num_to_input(
            incoords=self.incoords,
            outcoords=self.outcoords,
            cen=self.cen,
            npx=self.npx,
            dTheta=self.dTheta,
            thetas=self.thetas,
        )

    def _define_theta_at_center_of_elements(self, dTheta, npx):
        thetas = dTheta * (np.arange((-(npx - 1) / 2), ((npx - 1) / 2) + 1))
        for n in np.arange(npx):
            thetas[n] = (n + 1) * dTheta

        thetas = thetas - np.mean(thetas)
        return thetas

    def _make_transducer_surface_map(self, nX, nY, cen, rad):
        convex_transmitter_map = ConvexTransmitterMap(
            cen=cen,
            rad=rad,
            num_x=nX,
            num_y=nY,
            ppw=self.ppw,
            material_properties=self.material_properties,
            simulation_params=self.simulation_params,
            rho0=self.rho0,
            beta0=self.beta0,
            c0=self.c0,
            a0=self.a0,
        )
        convex_receiver_map = ConvexReceiverMap(
            num_x=nX,
            num_y=nY,
            in_map=convex_transmitter_map.in_map,
            ppw=self.ppw,
            material_properties=self.material_properties,
            simulation_params=self.simulation_params,
        )
        return convex_transmitter_map, convex_receiver_map

    def _assign_transducer_num_to_input(self, incoords, outcoords, cen, npx, dTheta, thetas):
        # Assign which transducer number is assigned to each input.
        thetas_in = np.arctan2(incoords[:, 0] - cen[0], incoords[:, 1] - cen[1])
        thetas_out = np.arctan2(outcoords[:, 0] - cen[0], outcoords[:, 1] - cen[1])

        outcoords2 = np.zeros((npx, 2))
        incoords2 = np.zeros((npx, 2))

        outcoords = np.append(outcoords, np.zeros((outcoords.shape[0], 2)), axis=1)
        outcoords[:, 2] = 1
        outcoords[:, 3] = 0

        incoords = np.append(incoords, np.zeros((incoords.shape[0], 2)), axis=1)
        incoords[:, 2] = 0
        incoords[:, 3] = 0

        for tt in range(npx):
            # find which incoords are assigned to tt
            _less_than_max = thetas_in < (thetas[tt] + dTheta / 2)
            _greater_than_min = thetas_in > (thetas[tt] - dTheta / 2)
            _id_theta = np.where(np.logical_and(_less_than_max, _greater_than_min))[0]

            incoords[_id_theta, 3] = tt + 1
            # find center of tt tx element - do each dim separate cause sometimes idtheta is just one value
            incoords2[tt, 0] = np.mean(incoords[_id_theta, 0])
            incoords2[tt, 1] = np.mean(incoords[_id_theta, 1])
            # find which outcoords are assigned to tt
            _less_than_max = thetas_out < (thetas[tt] + dTheta / 2)
            _greater_than_min = thetas_out > (thetas[tt] - dTheta / 2)
            _id_theta = np.where(np.logical_and(_less_than_max, _greater_than_min))[0]
            outcoords[_id_theta, 3] = tt + 1
            # find center of tt rx element - do each dim separate cause sometimes
            outcoords2[tt, 0] = np.mean(outcoords[_id_theta, 0])
            outcoords2[tt, 1] = np.mean(outcoords[_id_theta, 1])

        nOutPx = thetas_out.shape[0]
        nInPx = thetas_in.shape[0]
        return thetas_in, thetas_out, incoords, outcoords, incoords2, outcoords2, nOutPx, nInPx
