import numpy as np

from fullwave_simulation.solvers.input_generator_base import InputGeneratorBase


# pylint: disable=C0103
class FSAInputGenerator(InputGeneratorBase):
    def run(
        self,
        i_event,
    ):
        simulation_dir = self._work_dir / f"{i_event + 1}"
        simulation_dir.mkdir(parents=True, exist_ok=True)

        initial_condition_mat = self.initial_condition.generate_icmat(i_event)

        self._write_ic(simulation_dir / "icmat.dat", np.transpose(initial_condition_mat))
        self._copy_simulation_bin_file(simulation_dir)

        if i_event == 0:
            x_start = self._obtain_x_start_attenuation_dispersion_curves()

            kappa_x, kappa_y, kappa_u, kappa_w = self._calc_kappa(x_start, self._A)

            (
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
            ) = self._calc_pml_boundary_conditions(x_start, A=self._A)

            (
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
            ) = self._calc_gradient_masks(
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

            self._save_variables_into_dat_file(
                simulation_dir.parent,
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
                dim=self._dim,
                is_fsa=True,
            )
        self._build_symbolic_links(simulation_dir)

        return simulation_dir

    def _set_field_params(self):
        lambda_ = self._c0 / self._omega0 * 2 * np.pi
        self._nX = self._c_map.shape[0]
        self._nY = self._c_map.shape[1]
        self._nXe = self._nX + 2 * (self._num_body + self._m)
        self._nYe = self._nY + 2 * (self._num_body + self._m)
        self._nT = np.round(self._duration * self._c0 / lambda_ * self._ppw / self._cfl)
        #  Number of time samples in output after downsampling
        self.nT2 = len(np.arange(0, self._nT, self.simulation_params.modT))
        self._nTic = self.initial_condition.nTic
