import abc
import os
import shutil
from pathlib import Path

import numpy as np

import fullwave_simulation
from fullwave_simulation.conditions import FSAInitialCondition
from fullwave_simulation.domains import geometry_utils
from fullwave_simulation.utils.utils import matlab_round


# pylint: disable=C0103
class InputGeneratorBase:
    def __init__(
        self,
        work_dir,
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition: FSAInitialCondition,
        path_fullwave_simulation_bin=(
            Path(fullwave_simulation.__file__).parent.parent
            / "fullwave_simulation/bin/fullwave2_try6_nln_relaxing_pzero_rebuild3"
        ),
        m=8,
        num_body=40,
    ):
        self._work_dir = Path(work_dir)
        self._path_fullwave_simulation_bin = path_fullwave_simulation_bin
        assert self._path_fullwave_simulation_bin.exists()

        self.simulation_params = simulation_params
        self.domain_organizer = domain_organizer

        self.transducer = transducer
        self.wave_transmitter = wave_transmitter
        self.signal_receiver = signal_receiver

        self.initial_condition = initial_condition

        self._m = m
        self._num_body = num_body
        self._c0 = self.simulation_params.c0
        self._omega0 = self.simulation_params.omega0
        self._duration = self.simulation_params.dur
        self._ppw = self.transducer.ppw
        self._cfl = self.simulation_params.cfl
        self._modT = self.simulation_params.modT
        self._c_map = domain_organizer.constructed_domain_dict["c_map"]
        self._rho_map = domain_organizer.constructed_domain_dict["rho_map"]
        self._A_map = domain_organizer.constructed_domain_dict["a_map"]
        self._beta_map = domain_organizer.constructed_domain_dict["beta_map"]
        self._input_coords_zero = geometry_utils.map_to_coordinates_matlab(
            domain_organizer.constructed_domain_dict["air_map"]
        )

        self._input_coords = self.transducer.incoords
        self._output_coords = self.transducer.outcoords

        self._r = self.simulation_params.cfl

        self._set_field_params()
        self._set_step_params()
        self._c, self._rho, self._beta, self._A, self._K = self._extend_input_maps(
            c_map=self._c_map, rho_map=self._rho_map, beta_map=self._beta_map, A_map=self._A_map
        )
        self._dim = int(matlab_round(self._c.max()) - matlab_round(self._c.min()))

        self._set_coords()
        self._set_d_mat()
        self._set_d_map(self._dim, self._c)
        self._set_dc_map(self._c)
        self._set_pmls_params()

    @abc.abstractmethod
    def run(
        self,
        i_event,
    ):
        raise NotImplementedError

    # --- constructor utils ---

    def _set_field_params(self):
        lambda_ = self._c0 / self._omega0 * 2 * np.pi
        self._nX = self._c_map.shape[0]
        self._nY = self._c_map.shape[1]
        self._nXe = self._nX + 2 * (self._num_body + self._m)
        self._nYe = self._nY + 2 * (self._num_body + self._m)
        self._nT = np.round(self._duration * self._c0 / lambda_ * self._ppw / self._cfl)
        #  Number of time samples in output after downsampling
        self.nT2 = len(np.arange(0, self._nT, self.simulation_params.modT))
        # self._nTic = self.initial_condition.nTic

    def _set_step_params(self):
        self._dX = self._c0 / self._omega0 * 2 * np.pi / self._ppw
        self._dY = self._c0 / self._omega0 * 2 * np.pi / self._ppw
        self._dT = self._dY / self._c0 * self._cfl

    def _extend_input_maps(self, c_map, rho_map, beta_map, A_map):
        c = self._extend_map(c_map, self._num_body + self._m)
        rho = self._extend_map(rho_map, self._num_body + self._m)
        beta = self._extend_map(beta_map, self._num_body + self._m)
        A = self._extend_map(A_map, self._num_body + self._m)
        K = np.multiply(c**2, rho)
        return c, rho, beta, A, K

    def _set_coords(self):
        self._n_coords = self._input_coords.shape[0]
        self._n_coords_out = self._output_coords.shape[0]
        self._n_coords_zero = self._input_coords_zero.shape[0]
        self._input_coords = self._input_coords + self._num_body + self._m
        self._output_coords = self._output_coords + self._num_body + self._m
        self._input_coords_zero = self._input_coords_zero + self._num_body + self._m

    def _set_d_mat(self):
        # For 2D modeling:
        self._d = np.zeros((9, 2))
        self._d[1, 0] = (
            -0.000874634088067635 * self._r**7
            - 0.00180530560296097 * self._r**6
            - 0.000440512972481673 * self._r**5
            + 0.00474018847663366 * self._r**4
            - 1.93097802254349e-05 * self._r**3
            - 0.292328221171893 * self._r**2
            - 6.58101498708345e-08 * self._r
            + 1.25420636437969
        )
        self._d[2, 0] = (
            0.000793317828964018 * self._r**7
            + 0.00161433256585486 * self._r**6
            + 0.000397244786277123 * self._r**5
            + 0.00546057645976549 * self._r**4
            + 1.73781972873916e-05 * self._r**3
            + 0.0588754971188371 * self._r**2
            + 5.91706982879834e-08 * self._r
            - 0.123406473759703
        )
        self._d[3, 0] = (
            -0.000650217700538851 * self._r**7
            - 0.00116449260340413 * self._r**6
            - 0.000324403734066325 * self._r**5
            - 0.00911483710059994 * self._r**4
            - 1.417399823126e-05 * self._r**3
            + 0.0233184077551615 * self._r**2
            - 4.82326094707544e-08 * self._r
            + 0.0346342451534453
        )
        self._d[4, 0] = (
            0.000467529510541428 * self._r**7
            + 0.000732736676632388 * self._r**6
            + 0.000232444388955328 * self._r**5
            + 0.00846419766685254 * self._r**4
            + 1.01438593426278e-05 * self._r**3
            - 0.0317586249260511 * self._r**2
            + 3.44988852042879e-08 * self._r
            - 0.0119674942518101
        )
        self._d[5, 0] = (
            -0.000298416281187033 * self._r**7
            - 0.000399380750669364 * self._r**6
            - 0.000148203388388213 * self._r**5
            - 0.00601788793192501 * self._r**4
            - 6.46543538517443e-06 * self._r**3
            + 0.0241912754935119 * self._r**2
            - 2.19855171569984e-08 * self._r
            + 0.00415554391204146
        )
        self._d[6, 0] = (
            0.000167882669698981 * self._r**7
            + 0.000188195874702691 * self._r**6
            + 8.3057921860396e-05 * self._r**5
            + 0.00348461963201376 * self._r**4
            + 3.61873162287129e-06 * self._r**3
            - 0.0149875789940005 * self._r**2
            + 1.22979142197165e-08 * self._r
            - 0.00129213888778954
        )
        self._d[7, 0] = (
            -6.22209937489143e-05 * self._r**7
            - 6.44890425871692e-05 * self._r**6
            - 3.02936928954918e-05 * self._r**5
            - 0.00133386143898282 * self._r**4
            - 1.31215186728213e-06 * self._r**3
            + 0.00670228205200379 * self._r**2
            - 4.44653967516776e-09 * self._r
            + 0.000315659916047599
        )
        self._d[8, 0] = (
            6.8474088109024e-06 * self._r**7
            + 1.14082245705934e-05 * self._r**6
            + 3.0372759370575e-06 * self._r**5
            + 0.000236122782444105 * self._r**4
            + 1.26768491232397e-07 * self._r**3
            - 0.00153347270556276 * self._r**2
            + 4.21617557752767e-10 * self._r
            - 4.51948990428065e-05
        )
        self._d[1, 1] = (
            2.13188763071246e-06 * self._r**7
            - 7.41025068776257e-05 * self._r**6
            + 2.31652037371554e-06 * self._r**5
            - 0.00259495924602038 * self._r**4
            + 1.20637183170338e-07 * self._r**3
            + 0.0521123771632193 * self._r**2
            + 4.42258843694177e-10 * self._r
            - 4.20967682664542e-07
        )

    def _set_d_map(self, dim, c):
        self._d_map = np.zeros((9, 2, dim + 1))
        for i in range(0, dim + 1):
            r_d_map = (i + c.min()) * self._dT / self._dX
            self._d_map[1, 0, i] = (
                -0.000874634088067635 * r_d_map**7
                - 0.00180530560296097 * r_d_map**6
                - 0.000440512972481673 * r_d_map**5
                + 0.00474018847663366 * r_d_map**4
                - 1.93097802254349e-05 * r_d_map**3
                - 0.292328221171893 * r_d_map**2
                - 6.58101498708345e-08 * r_d_map
                + 1.25420636437969
            )
            self._d_map[2, 0, i] = (
                0.000793317828964018 * r_d_map**7
                + 0.00161433256585486 * r_d_map**6
                + 0.000397244786277123 * r_d_map**5
                + 0.00546057645976549 * r_d_map**4
                + 1.73781972873916e-05 * r_d_map**3
                + 0.0588754971188371 * r_d_map**2
                + 5.91706982879834e-08 * r_d_map
                - 0.123406473759703
            )
            self._d_map[3, 0, i] = (
                -0.000650217700538851 * r_d_map**7
                - 0.00116449260340413 * r_d_map**6
                - 0.000324403734066325 * r_d_map**5
                - 0.00911483710059994 * r_d_map**4
                - 1.417399823126e-05 * r_d_map**3
                + 0.0233184077551615 * r_d_map**2
                - 4.82326094707544e-08 * r_d_map
                + 0.0346342451534453
            )
            self._d_map[4, 0, i] = (
                0.000467529510541428 * r_d_map**7
                + 0.000732736676632388 * r_d_map**6
                + 0.000232444388955328 * r_d_map**5
                + 0.00846419766685254 * r_d_map**4
                + 1.01438593426278e-05 * r_d_map**3
                - 0.0317586249260511 * r_d_map**2
                + 3.44988852042879e-08 * r_d_map
                - 0.0119674942518101
            )
            self._d_map[5, 0, i] = (
                -0.000298416281187033 * r_d_map**7
                - 0.000399380750669364 * r_d_map**6
                - 0.000148203388388213 * r_d_map**5
                - 0.00601788793192501 * r_d_map**4
                - 6.46543538517443e-06 * r_d_map**3
                + 0.0241912754935119 * r_d_map**2
                - 2.19855171569984e-08 * r_d_map
                + 0.00415554391204146
            )
            self._d_map[6, 0, i] = (
                0.000167882669698981 * r_d_map**7
                + 0.000188195874702691 * r_d_map**6
                + 8.3057921860396e-05 * r_d_map**5
                + 0.00348461963201376 * r_d_map**4
                + 3.61873162287129e-06 * r_d_map**3
                - 0.0149875789940005 * r_d_map**2
                + 1.22979142197165e-08 * r_d_map
                - 0.00129213888778954
            )
            self._d_map[7, 0, i] = (
                -6.22209937489143e-05 * r_d_map**7
                - 6.44890425871692e-05 * r_d_map**6
                - 3.02936928954918e-05 * r_d_map**5
                - 0.00133386143898282 * r_d_map**4
                - 1.31215186728213e-06 * r_d_map**3
                + 0.00670228205200379 * r_d_map**2
                - 4.44653967516776e-09 * r_d_map
                + 0.000315659916047599
            )
            self._d_map[8, 0, i] = (
                6.8474088109024e-06 * r_d_map**7
                + 1.14082245705934e-05 * r_d_map**6
                + 3.0372759370575e-06 * r_d_map**5
                + 0.000236122782444105 * r_d_map**4
                + 1.26768491232397e-07 * r_d_map**3
                - 0.00153347270556276 * r_d_map**2
                + 4.21617557752767e-10 * r_d_map
                - 4.51948990428065e-05
            )
            self._d_map[1, 1, i] = (
                2.13188763071246e-06 * r_d_map**7
                - 7.41025068776257e-05 * r_d_map**6
                + 2.31652037371554e-06 * r_d_map**5
                - 0.00259495924602038 * r_d_map**4
                + 1.20637183170338e-07 * r_d_map**3
                + 0.0521123771632193 * r_d_map**2
                + 4.42258843694177e-10 * r_d_map
                - 4.20967682664542e-07
            )

    def _set_dc_map(self, c):
        self._dc_map = matlab_round(c) - matlab_round(c.min()) + 1

    def _set_pmls_params(self):
        self._kpml = 1
        self._L = self._dX * self._num_body
        self._Rc = 1e-30
        self._d0 = -3 * self._c0 * np.log(self._Rc) / (2 * self._L)

    def _extend_map(self, input_map, num_boundary_points):
        output_map = np.zeros(
            (
                input_map.shape[0] + 2 * num_boundary_points,
                input_map.shape[1] + 2 * num_boundary_points,
            )
        )

        # center
        output_map[
            num_boundary_points:-num_boundary_points, num_boundary_points:-num_boundary_points
        ] = input_map

        # edges
        for i in range(input_map.shape[0]):
            output_map[i + num_boundary_points, :num_boundary_points] = (
                np.ones(num_boundary_points) * input_map[i, 0]
            )
            output_map[i + num_boundary_points, -num_boundary_points:] = (
                np.ones(num_boundary_points) * input_map[i, -1]
            )
        for j in range(input_map.shape[1]):
            output_map[:num_boundary_points, j + num_boundary_points] = (
                np.ones(num_boundary_points) * input_map[0, j]
            )
            output_map[-num_boundary_points:, j + num_boundary_points] = (
                np.ones(num_boundary_points) * input_map[-1, j]
            )

        # corners
        output_map[:num_boundary_points, :num_boundary_points] = input_map[0, 0]
        output_map[-num_boundary_points:, -num_boundary_points:] = input_map[-1, -1]
        output_map[-num_boundary_points:, :num_boundary_points] = input_map[-1, 0]
        output_map[:num_boundary_points, -num_boundary_points:] = input_map[0, -1]

        return output_map

    # --- runner utils ---

    def _calc_pml_boundary_conditions(self, x_start, A):
        _d_pml_x_old = self._calc_d_pml_x_old()
        _alpha_pml_x_old = self._calc_alpha_pml_x_Old()
        a_pml_x_old, b_pml_x_old = self._calc_a_and_b(
            _d_pml_x_old, self._kpml, _alpha_pml_x_old, self._dT
        )

        _d_pml_x_old2 = self._calc_d_pml_x_old2()
        _alpha_pml_x_old2 = self._calc_alpha_pml_x_Old2()
        a_pml_x_old2, b_pml_x_old2 = self._calc_a_and_b(
            _d_pml_x_old2, self._kpml, _alpha_pml_x_old2, self._dT
        )

        _d_pml_y_old = self._calc_d_pml_y_old()
        _alpha_pml_y_old = self._calc_alpha_pml_y_old()
        a_pml_y_old, b_pml_y_old = self._calc_a_and_b(
            _d_pml_y_old, self._kpml, _alpha_pml_y_old, self._dT
        )

        _d_pml_y_old2 = self._calc_d_pml_y_old2()
        _alpha_pml_y_old2 = self._calc_alpha_pml_y_old2()
        a_pml_y_old2, b_pml_y_old2 = self._calc_a_and_b(
            _d_pml_y_old2, self._kpml, _alpha_pml_y_old2, self._dT
        )

        a_pml_x1, b_pml_x1 = self._calc_a_and_b(
            x_start[1], x_start[0], x_start[2], self._dT / A * 0.5
        )
        a_pml_x2, b_pml_x2 = self._calc_a_and_b(
            x_start[3], x_start[0], x_start[4], self._dT / A * 0.5
        )
        a_pml_u1, b_pml_u1 = self._calc_a_and_b(
            x_start[6], x_start[5], x_start[7], self._dT / A * 0.5
        )
        a_pml_u2, b_pml_u2 = self._calc_a_and_b(
            x_start[8], x_start[5], x_start[9], self._dT / A * 0.5
        )
        return (
            a_pml_u1,
            b_pml_u1,
            a_pml_x1,
            b_pml_x1,
            a_pml_x2,
            b_pml_x2,
            a_pml_u2,
            b_pml_u2,
            #
            a_pml_x_old,
            a_pml_x_old2,
            a_pml_y_old,
            a_pml_y_old2,
            b_pml_x_old,
            b_pml_x_old2,
            b_pml_y_old,
            b_pml_y_old2,
        )

    def _calc_gradient_masks(
        self,
        a_pml_u1,
        b_pml_u1,
        a_pml_x1,
        b_pml_x1,
        a_pml_x2,
        b_pml_x2,
        a_pml_u2,
        b_pml_u2,
        #
        a_pml_x_Old,
        a_pml_x_Old2,
        a_pml_y_Old,
        a_pml_y_Old2,
        b_pml_x_Old,
        b_pml_x_Old2,
        b_pml_y_Old,
        b_pml_y_Old2,
    ):
        (
            a_pml_y1,
            b_pml_y1,
            a_pml_y2,
            b_pml_y2,
            a_pml_w1,
            b_pml_w1,
            a_pml_w2,
            b_pml_w2,
        ) = self._coords_transformation(
            a_pml_x1,
            b_pml_x1,
            a_pml_x2,
            b_pml_x2,
            a_pml_u1,
            b_pml_u1,
            a_pml_u2,
            b_pml_u2,
        )
        # pmlmaskx, pmlmasky = self._localize_pml_region()
        a_pml_x1, a_pml_u1 = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body / 2,
            self._m + self._num_body / 2,
            1,
            a_pml_x1,
            a_pml_u1,
        )
        a_pml_x2, a_pml_u2 = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body / 2,
            self._m + self._num_body / 2,
            1,
            a_pml_x2,
            a_pml_u2,
        )
        a_pml_y1, a_pml_w1 = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body / 2,
            self._m + self._num_body / 2,
            2,
            a_pml_y1,
            a_pml_w1,
        )
        a_pml_y2, a_pml_w2 = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body / 2,
            self._m + self._num_body / 2,
            2,
            a_pml_y2,
            a_pml_w2,
        )
        b_pml_x1, b_pml_u1 = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body / 2,
            self._m + self._num_body / 2,
            1,
            b_pml_x1,
            b_pml_u1,
        )
        b_pml_x2, b_pml_u2 = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body / 2,
            self._m + self._num_body / 2,
            1,
            b_pml_x2,
            b_pml_u2,
        )
        b_pml_y1, b_pml_w1 = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body / 2,
            self._m + self._num_body / 2,
            2,
            b_pml_y1,
            b_pml_w1,
        )
        b_pml_y2, b_pml_w2 = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body / 2,
            self._m + self._num_body / 2,
            2,
            b_pml_y2,
            b_pml_w2,
        )
        a_pml_x1, _ = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body,
            self._m,
            1,
            a_pml_x1,
            np.transpose(a_pml_x_Old) * np.ones((1, self._nYe)),
        )
        a_pml_u1, _ = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body,
            self._m,
            1,
            a_pml_u1,
            np.transpose(a_pml_x_Old2) * np.ones((1, self._nYe)),
        )
        a_pml_y1, _ = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body,
            self._m,
            2,
            a_pml_y1,
            np.ones((self._nXe, 1)) * a_pml_y_Old,
        )
        a_pml_w1, _ = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body,
            self._m,
            2,
            a_pml_w1,
            np.ones((self._nXe, 1)) * a_pml_y_Old2,
        )
        b_pml_x1, _ = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body,
            self._m,
            1,
            b_pml_x1,
            np.transpose(b_pml_x_Old) * np.ones((1, self._nYe)),
        )
        b_pml_u1, _ = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body,
            self._m,
            1,
            b_pml_u1,
            np.transpose(b_pml_x_Old2) * np.ones((1, self._nYe)),
        )
        b_pml_y1, _ = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body,
            self._m,
            2,
            b_pml_y1,
            np.ones((self._nXe, 1)) * b_pml_y_Old,
        )
        b_pml_w1, _ = self._pml_gradient_mask2(
            self._nXe,
            self._nYe,
            self._num_body,
            self._m,
            2,
            b_pml_w1,
            np.ones((self._nXe, 1)) * b_pml_y_Old2,
        )
        return (
            a_pml_u1,
            b_pml_u1,
            a_pml_w1,
            b_pml_w1,
            #
            a_pml_x1,
            b_pml_x1,
            a_pml_y1,
            b_pml_y1,
            #
            a_pml_u2,
            b_pml_u2,
            a_pml_w2,
            b_pml_w2,
            #
            a_pml_x2,
            b_pml_x2,
            a_pml_y2,
            b_pml_y2,
        )

    def _localize_pml_region(self):
        pml_mask_x = np.zeros((self._nXe, self._nYe))
        pml_mask_y = np.zeros((self._nXe, self._nYe))

        for i in range(self._num_body):
            pml_mask_x[i + (self._nXe - self._m - self._num_body + 1), :] = i / self._num_body
            pml_mask_x[self._m + self._num_body + 1 - i, :] = i / self._num_body
            pml_mask_y[:, i + (self._nYe - self._m - self._num_body + 1)] = i / self._num_body
            pml_mask_y[:, self._m + self._num_body + 1 - i] = i / self._num_body

        pml_mask_x[0 : self._m, :] = 1
        pml_mask_x[self._nXe - self._m : self._nXe + 1, :] = 1

        pml_mask_y[:, 0 : self._m + 1] = 1
        pml_mask_y[:, self._nYe - self._m : self._nYe + 1] = 1
        return pml_mask_x, pml_mask_y

    def _coords_transformation(
        self,
        a_pml_x1,
        b_pml_x1,
        a_pml_x2,
        b_pml_x2,
        a_pml_u1,
        b_pml_u1,
        a_pml_u2,
        b_pml_u2,
    ):
        a_pml_y1 = a_pml_x1.copy()
        b_pml_y1 = b_pml_x1.copy()
        a_pml_y2 = a_pml_x2.copy()
        b_pml_y2 = b_pml_x2.copy()
        a_pml_w1 = a_pml_u1.copy()
        b_pml_w1 = b_pml_u1.copy()
        a_pml_w2 = a_pml_u2.copy()
        b_pml_w2 = b_pml_u2.copy()
        return a_pml_y1, b_pml_y1, a_pml_y2, b_pml_y2, a_pml_w1, b_pml_w1, a_pml_w2, b_pml_w2

    def _calc_kappa(self, xstart, A):
        kappa_x = np.zeros((self._nXe, self._nYe)) + 1 - (1 - xstart[0]) * np.sqrt(A / 0.5)
        kappa_y = np.zeros((self._nXe, self._nYe)) + 1 - (1 - xstart[0]) * np.sqrt(A / 0.5)
        kappa_u = np.zeros((self._nXe, self._nYe)) + 1 - (1 - xstart[5]) * np.sqrt(A / 0.5)
        kappa_w = np.zeros((self._nXe, self._nYe)) + 1 - (1 - xstart[5]) * np.sqrt(A / 0.5)
        return kappa_x, kappa_y, kappa_u, kappa_w

    def _obtain_x_start_attenuation_dispersion_curves(self):
        x_start = np.array(
            [
                0.987800000000000,
                17824.8199986552,
                2123790.42009930,
                6293321.78584677,
                1271128233.91059,
                0.989000000000000,
                99976.9860944360,
                21223237.1761988,
                12580851.0728108,
                1143364660.54746,
            ]
        )
        return x_start

    def _calc_alpha_pml_y_old2(self):
        alpha_pml_y_Old2 = np.zeros((1, self._nYe))

        for i in range(self._num_body):
            alpha_pml_y_Old2[:, i + (self._nYe - self._m - self._num_body + 1)] = (
                self._omega0 / 2 * (self._num_body - ((i + 1) - 1 / 2)) / self._num_body
            )
            alpha_pml_y_Old2[:, self._m + self._num_body + 1 - (i + 2) - 1] = (
                self._omega0 / 2 * (self._num_body - ((i + 1) - 1 / 2)) / self._num_body
            )
        alpha_pml_y_Old2 = alpha_pml_y_Old2 / 10
        return alpha_pml_y_Old2

    def _calc_d_pml_y_old2(self):
        d_pml_y_Old2 = np.zeros((1, self._nYe))

        for i in range(self._num_body):
            d_pml_y_Old2[:, i + (self._nYe - self._m - self._num_body + 1)] = (
                self._d0 * (((i + 1) - 1 / 2) / self._num_body) ** 2
            )
            d_pml_y_Old2[:, self._m + self._num_body + 1 - (i + 2) - 1] = (
                self._d0 * (((i + 1) - 1 / 2) / self._num_body) ** 2
            )
        return d_pml_y_Old2

    def _calc_alpha_pml_y_old(self):
        alpha_pml_y_Old = np.zeros((1, self._nYe))
        for i in range(self._num_body):
            alpha_pml_y_Old[:, i + (self._nYe - self._m - self._num_body + 1)] = (
                self._omega0 / 2 * (self._num_body - (i + 1)) / self._num_body
            )
            alpha_pml_y_Old[:, self._m + self._num_body + 1 - (i + 2)] = (
                self._omega0 / 2 * (self._num_body - (i + 1)) / self._num_body
            )
        alpha_pml_y_Old = alpha_pml_y_Old / 10
        return alpha_pml_y_Old

    def _calc_d_pml_y_old(self):
        d_pml_y_Old = np.zeros((1, self._nYe))
        for i in range(self._num_body):
            d_pml_y_Old[:, i + (self._nYe - self._m - self._num_body + 1)] = (
                self._d0 * ((i + 1) / self._num_body) ** 2
            )
            d_pml_y_Old[:, self._m + self._num_body + 1 - (i + 2)] = (
                self._d0 * ((i + 1) / self._num_body) ** 2
            )
        return d_pml_y_Old

    def _calc_d_pml_x_old(self):
        d_pml_x_Old = np.zeros((1, self._nXe))
        for i in range(self._num_body):
            d_pml_x_Old[:, i + (self._nXe - self._m - self._num_body + 1)] = (
                self._d0 * ((i + 1) / self._num_body) ** 2
            )
            d_pml_x_Old[:, self._m + self._num_body + 1 - (i + 2)] = (
                self._d0 * ((i + 1) / self._num_body) ** 2
            )
        return d_pml_x_Old

    def _calc_alpha_pml_x_Old(self):
        alpha_pml_x_Old = np.zeros((1, self._nXe))
        for i in range(self._num_body):
            alpha_pml_x_Old[:, i + (self._nXe - self._m - self._num_body + 1)] = (
                self._omega0 / 2 * (self._num_body - (i + 1)) / self._num_body
            )
            alpha_pml_x_Old[:, self._m + self._num_body + 1 - (i + 2)] = (
                self._omega0 / 2 * (self._num_body - (i + 1)) / self._num_body
            )
        alpha_pml_x_Old = alpha_pml_x_Old / 10  # ToDo: why / 10?
        return alpha_pml_x_Old

    def _calc_d_pml_x_old2(self):
        d_pml_x_Old2 = np.zeros((1, self._nXe))
        for i in range(self._num_body):
            d_pml_x_Old2[:, i + (self._nXe - self._m - self._num_body + 1)] = (
                self._d0 * (((i + 1) - 1 / 2) / self._num_body) ** 2
            )
            d_pml_x_Old2[:, self._m + self._num_body + 1 - (i + 2) - 1] = (
                self._d0 * (((i + 1) - 1 / 2) / self._num_body) ** 2
            )
        return d_pml_x_Old2

    def _calc_alpha_pml_x_Old2(self):
        alpha_pml_x_Old2 = np.zeros((1, self._nXe))
        for i in range(self._num_body):
            alpha_pml_x_Old2[:, i + (self._nXe - self._m - self._num_body + 1)] = (
                self._omega0 / 2 * (self._num_body - ((i + 1) - 1 / 2)) / self._num_body
            )
            alpha_pml_x_Old2[:, self._m + self._num_body + 1 - (i + 2) - 1] = (
                self._omega0 / 2 * (self._num_body - ((i + 1) - 1 / 2)) / self._num_body
            )
        alpha_pml_x_Old2 = alpha_pml_x_Old2 / 10
        return alpha_pml_x_Old2

    # def _pml_gradient_mask(
    #     self,
    #     nX=None,
    #     nY=None,
    #     nbdy=None,
    #     M=None,
    #     crd=None,
    #     apmlx1=None,
    #     apmlu1=None,
    # ):
    #     nbdy = int(nbdy)
    #     M = int(M)
    #     pmlmask = np.zeros((nX, nY))
    #     if crd == 1:
    #         # for i in np.arange(1, nbdy + 1).reshape(-1):
    #         for i in range(nbdy):
    #             pmlmask[i + (nX - M - nbdy + 1), :] = i / nbdy
    #             pmlmask[M + nbdy + 1 - i, :] = i / nbdy
    #         pmlmask[0 : M + 1, :] = 1
    #         pmlmask[nX - M : nX + 1, :] = 1

    #     if crd == 2:
    #         pmlmask = np.zeros((nX, nY))
    #         # for i in np.arange(1, nbdy + 1).reshape(-1):
    #         for i in range(nbdy):
    #             pmlmask[:, i + (nY - M - nbdy + 1)] = i / nbdy
    #             pmlmask[:, M + nbdy + 1 - i] = i / nbdy
    #         pmlmask[:, 0 : M + 1] = 1
    #         pmlmask[:, nY - M : nY + 1, :] = 1

    #     tmpx = apmlx1
    #     tmpu = apmlu1
    #     apmlx1 = np.multiply(tmpx, (1 - pmlmask)) + np.multiply((tmpx + tmpu) / 2, pmlmask)
    #     apmlu1 = np.multiply(tmpu, (1 - pmlmask)) + np.multiply((tmpx + tmpu) / 2, pmlmask)
    #     return apmlx1, apmlu1

    def _pml_gradient_mask2(
        self,
        nX,
        nY,
        num_body,
        M,
        crd,
        a_pml_x1,
        a_pml_u1,
    ):
        num_body = int(num_body)
        M = int(M)
        pml_mask = np.zeros((nX, nY))
        if crd == 1:
            # for i in np.arange(1, nbdy + 1).reshape(-1):
            for i in range(int(num_body)):
                pml_mask[i + (nX - M - num_body + 1), :] = (i + 1) / num_body
                pml_mask[M + num_body + 1 - i - 2, :] = (i + 1) / num_body
            pml_mask[0:M, :] = 1
            pml_mask[nX - M : nX, :] = 1

        if crd == 2:
            # for i in np.arange(1, nbdy + 1).reshape(-1):
            for i in range(num_body):
                pml_mask[:, i + (nY - M - num_body + 1)] = (i + 1) / num_body
                pml_mask[:, M + num_body + 1 - i - 2] = (i + 1) / num_body
            pml_mask[:, 0:M] = 1
            pml_mask[:, nY - M : nY] = 1

        tmpx = a_pml_x1.copy()
        tmpu = a_pml_u1.copy()
        a_pml_x1 = np.multiply(tmpx, (1 - pml_mask)) + np.multiply(tmpu, pml_mask)
        a_pml_u1 = np.multiply(tmpu, (1 - pml_mask)) + np.multiply(tmpx, pml_mask)
        return a_pml_x1, a_pml_u1

    def _calc_a_and_b(self, dx, kappa_x, alpha_x, dT):
        # function [a b] = ab(dx,kappax,alphax,dT)
        if dx.ndim >= 1:
            b = np.exp(-(dx / kappa_x + alpha_x) * dT)
            # why do I get scalar value when I do this on Matlab?
            # (kappax * (dx + kappax * alphax)) is a vector
            # dx is also a vector.
            # However, I get a scalar value when I do right division on Matlab.
            temp = np.linalg.lstsq((kappa_x * (dx + kappa_x * alpha_x)).T, dx.T)[0][0][0]
            a = temp * (b - 1)
        else:
            b = np.exp(-(dx / kappa_x + alpha_x) * dT)
            a = dx / (kappa_x * (dx + kappa_x * alpha_x)) * (b - 1)
        return a, b

    # --- saving utils ---

    def _save_variables_into_dat_file(
        self,
        simulation_dir,
        kappa_x,
        kappa_y,
        kappa_u,
        kappa_w,
        a_pml_u1,
        b_pml_u1,
        a_pml_w1,
        b_pml_w1,
        a_pml_x1,
        b_pml_x1,
        a_pml_y1,
        b_pml_y1,
        a_pml_u2,
        b_pml_u2,
        a_pml_w2,
        b_pml_w2,
        a_pml_x2,
        b_pml_x2,
        a_pml_y2,
        b_pml_y2,
        dim,
        is_fsa=False,
    ):
        if is_fsa:
            self._save_maps(
                simulation_dir,
                self._c,
                self._K,
                self._rho,
                self._beta,
            )
        self._save_coords(simulation_dir=simulation_dir)
        self._save_step_params(simulation_dir)
        self._save_coords_params(simulation_dir)
        self._save_d_params(simulation_dir, dim)

        # save kappa and pml params
        for var_name, var in [
            ("kappax", kappa_x),
            ("kappay", kappa_y),
            ("kappau", kappa_u),
            ("kappaw", kappa_w),
            ("apmlu1", a_pml_u1),
            ("bpmlu1", b_pml_u1),
            ("apmlw1", a_pml_w1),
            ("bpmlw1", b_pml_w1),
            ("apmlx1", a_pml_x1),
            ("bpmlx1", b_pml_x1),
            ("apmly1", a_pml_y1),
            ("bpmly1", b_pml_y1),
            ("apmlu2", a_pml_u2),
            ("bpmlu2", b_pml_u2),
            ("apmlw2", a_pml_w2),
            ("bpmlw2", b_pml_w2),
            ("apmlx2", a_pml_x2),
            ("bpmlx2", b_pml_x2),
            ("apmly2", a_pml_y2),
            ("bpmly2", b_pml_y2),
        ]:
            save_path = simulation_dir / f"{var_name}.dat"
            self._write_matrix(var_type=np.float32, save_path=save_path, variable_mat=var)

    def _build_symbolic_links(self, simulation_dir):
        for var_name in [
            "c",
            "K",
            "rho",
            "beta",
            #
            "dX",
            "dY",
            "dT",
            "c0",
            #
            "icc",
            "outc",
            #
            "nY",
            "nX",
            "nT",
            "ncoords",
            "ncoordsout",
            "nTic",
            "modT",
            #
            "d",
            "dmap",
            "ndmap",
            "dcmap",
            #
            "kappax",
            "kappay",
            "kappau",
            "kappaw",
            "apmlu1",
            "bpmlu1",
            "apmlw1",
            "bpmlw1",
            "apmlx1",
            "bpmlx1",
            "apmly1",
            "bpmly1",
            "apmlu2",
            "bpmlu2",
            "apmlw2",
            "bpmlw2",
            "apmlx2",
            "bpmlx2",
            "apmly2",
            "bpmly2",
        ]:
            src_data = simulation_dir.parent / f"{var_name}.dat"
            target_data = simulation_dir / f"{var_name}.dat"
            # generate the symlink even if the file already exists
            if target_data.exists():
                target_data.unlink()
            os.symlink(src_data, target_data)

    def _save_maps(
        self,
        simulation_dir,
        c,
        K,
        rho,
        beta,
    ):
        c.astype(np.float32).tofile(simulation_dir / "c.dat")
        K.astype(np.float32).tofile(simulation_dir / "K.dat")
        rho.astype(np.float32).tofile(simulation_dir / "rho.dat")
        beta.astype(np.float32).tofile(simulation_dir / "beta.dat")

    def _save_coords(self, simulation_dir):
        self._write_coords(simulation_dir / "icc.dat", self._input_coords - 1)
        self._write_coords(simulation_dir / "outc.dat", self._output_coords - 1)
        self._write_coords(simulation_dir / "icczero.dat", self._input_coords_zero - 1)

        # self._write_ic(simulation_dir / "icmat.dat", np.transpose(initial_condition_mat))

    def _save_step_params(self, simulation_dir):
        for var_name, var in [
            ("dX", self._dX),
            ("dY", self._dY),
            ("dT", self._dT),
            ("c0", self._c0),
        ]:
            save_path = simulation_dir / f"{var_name}.dat"
            self._write_v_abs(np.float32, save_path, var)

    def _save_coords_params(self, simulation_dir):
        for var_name, var in [
            ("nX", self._nXe),
            ("nY", self._nYe),
            ("nT", self._nT),
            ("ncoords", self._n_coords),
            ("ncoordsout", self._n_coords_out),
            ("ncoordszero", self._n_coords_zero),
            ("nTic", self._nTic),
            ("modT", self._modT),
        ]:
            save_path = simulation_dir / f"{var_name}.dat"
            self._write_v_abs(np.int32, save_path, var)

    def _save_d_params(self, simulation_dir, dim):
        # save d and dmap
        self._write_matrix(np.float32, simulation_dir / "d.dat", self._d)
        self._write_matrix(np.float32, simulation_dir / "dmap.dat", self._d_map)

        # save ndmap
        if dim == 0:
            ndmap = 1
        else:
            ndmap = self._d_map.shape[2]
        self._write_v_abs(np.int32, simulation_dir / "ndmap.dat", ndmap)

        # save dcmap
        self._write_matrix(
            var_type=np.int32,
            save_path=simulation_dir / "dcmap.dat",
            variable_mat=(self._dc_map - 1),
            # variable_mat=(self._dc_map),
        )

    def _copy_simulation_bin_file(self, simulation_dir):
        shutil.copy(
            src=self._path_fullwave_simulation_bin,
            dst=simulation_dir / self._path_fullwave_simulation_bin.name,
        )

    def _write_ic(self, fname, icmat):
        icmat.T.flatten().astype(np.float32).tofile(fname)

    def _write_coords(self, fname, coords):
        coords.T.flatten().astype(np.int32).tofile(fname)

    def _write_v_abs(self, var_type, save_path, variable):
        np.array(variable).astype(var_type).tofile(save_path)

    def _write_matrix(self, var_type, save_path, variable_mat):
        variable_mat.astype(var_type).tofile(save_path)
