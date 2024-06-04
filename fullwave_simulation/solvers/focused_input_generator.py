from pathlib import Path

import numpy as np

import fullwave_simulation
from fullwave_simulation.domains import geometry_utils
from fullwave_simulation.solvers.input_generator_base import InputGeneratorBase
from fullwave_simulation.utils.utils import matlab_round


# pylint: disable=C0103
class FocusedInputGenerator(InputGeneratorBase):
    def __init__(
        self,
        work_dir,
        simulation_params,
        domain_organizer,
        transducer,
        wave_transmitter,
        signal_receiver,
        initial_condition,
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
        self._air_map = domain_organizer.constructed_domain_dict["air_map"]

        self._input_coords = self.transducer.incoords
        self._output_coords = self.transducer.outcoords

        self._r = self.simulation_params.cfl

        self._set_field_params()
        self._set_step_params()
        # self._extend_input_maps()
        self._set_coords()
        self._set_d_mat()
        self._set_pmls_params()

    def _set_field_params(self):
        lambda_ = self._c0 / self._omega0 * 2 * np.pi
        self._nX = self.transducer.num_x
        self._nY = self.transducer.num_y
        self._nXe = self._nX + 2 * (self._num_body + self._m)
        self._nYe = self._nY + 2 * (self._num_body + self._m)
        self._nT = np.round(self._duration * self._c0 / lambda_ * self._ppw / self._cfl)
        #  Number of time samples in output after downsampling
        self.nT2 = len(np.arange(0, self._nT, self.simulation_params.modT))
        # self._nTic = self.initial_condition.nTic

    def _set_coords(self):
        self._n_coords = self._input_coords.shape[0]
        self._n_coords_out = self._output_coords.shape[0]
        # self._n_coords_zero = self._input_coords_zero.shape[0]
        self._input_coords_org = self._input_coords.copy()
        self._output_coords_org = self._output_coords.copy()
        self._input_coords = self._input_coords + self._num_body + self._m
        self._output_coords = self._output_coords + self._num_body + self._m
        # self._input_coords_zero = self._input_coords_zero + self._num_body + self._m

    def _save_coords_params(self, simulation_dir):
        for var_name, var in [
            ("nX", self._nXe),
            ("nY", self._nYe),
            ("nT", self._nT),
            ("ncoords", self._n_coords),
            ("ncoordsout", self._n_coords_out),
            ("ncoordszero", self._n_coords_zero),
            ("nTic", self._nTic),
            # ("modT", self._modT),
        ]:
            save_path = simulation_dir / f"{var_name}.dat"
            self._write_v_abs(np.int32, save_path, var)

    def _cut_domain_for_scan(self, i_event):
        orig = int(
            np.round(self._c_map.shape[0] / 2)
            - np.round(self._nX / 2)
            - np.round(
                ((i_event + 1) - (self.simulation_params.nevents + 1) / 2)
                * self.transducer.beam_spacing
            )
            - 6
        )
        c = self._c_map[orig : orig + self._nX, 0 : self._nY]
        rho = self._rho_map[orig : orig + self._nX, 0 : self._nY]
        A = self._A_map[orig : orig + self._nX, 0 : self._nY]
        beta = self._beta_map[orig : orig + self._nX, 0 : self._nY]
        air = self._air_map[orig : orig + self._nX, 0 : self._nY]
        return c, rho, beta, A, air, orig

    def run(
        self,
        i_event,
    ):
        simulation_dir = self._work_dir / f"{i_event + 1}"
        simulation_dir.mkdir(parents=True, exist_ok=True)

        initial_condition_mat = self.initial_condition.generate_icmat()
        self._nTic = initial_condition_mat.shape[1]

        self._write_ic(simulation_dir / "icmat.dat", np.transpose(initial_condition_mat))
        self._copy_simulation_bin_file(simulation_dir)

        c, rho, beta, A, air, _ = self._cut_domain_for_scan(i_event)

        input_coords_zero = geometry_utils.map_to_coordinates_matlab(air)
        self._n_coords_zero = input_coords_zero.shape[0]
        self._input_coords_zero = input_coords_zero + self._num_body + self._m

        c, rho, beta, A, K = self._extend_input_maps(c_map=c, rho_map=rho, beta_map=beta, A_map=A)
        self._save_maps(simulation_dir=simulation_dir, c=c, K=K, rho=rho, beta=beta)
        dim = int(matlab_round(c.max()) - matlab_round(c.min()))
        self._set_d_map(dim, c)
        self._set_dc_map(c)

        x_start = self._obtain_x_start_attenuation_dispersion_curves()

        kappa_x, kappa_y, kappa_u, kappa_w = self._calc_kappa(x_start, A)

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
        ) = self._calc_pml_boundary_conditions(x_start, A)

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
        )

        return simulation_dir
