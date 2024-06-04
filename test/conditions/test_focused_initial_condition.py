from pathlib import Path

import numpy as np

from fullwave_simulation.conditions import FocusedInitialCondition
from fullwave_simulation.constants import FocusedSimulationParams, MaterialProperties
from fullwave_simulation.transducers import L125Transducer, LinearTxWaveTransmitter
from fullwave_simulation.utils import test_utils, utils


class InitialConditionMod(FocusedInitialCondition):
    def generate_icmat(self):
        test_data_dir = test_utils.get_test_data_dir("transducers")
        var_name = "icmat"

        icmat = np.zeros((self.transducer.incoords.shape[0], self._nT))
        icvec = self.wave_transmitter.generate_transmit_pulse(i=0)

        mat_file_path = test_data_dir / "linear_tx_focus_coords_coords_t=1.mat"
        test_utils.check_variable(
            mat_file_path=mat_file_path,
            var_name="coords",
            test_value=self.incoords[0 : int(self.incoords.shape[0] / 3), :],
        )

        icmat[0 : int(self.incoords.shape[0] / 3), :] = self._focus_coords(
            self.fcen[0],
            self.fcen[1],
            self.incoords[0 : int(self.incoords.shape[0] / 3), :],
            icvec,
            self.simulation_params.cfl,
        )
        mat_file_path = test_data_dir / f"linear_tx_icmat_t=1.mat"
        test_utils.check_variable(mat_file_path=mat_file_path, var_name=var_name, test_value=icmat)

        icvec = self.wave_transmitter.generate_transmit_pulse(i=1)
        icmat[
            int(self.incoords.shape[0] / 3) : int(self.incoords.shape[0] / 3 * 2), :
        ] = self._focus_coords(
            self.fcen[0],
            self.fcen[1],
            self.incoords[
                int(self.incoords.shape[0] / 3) : int(self.incoords.shape[0] / 3 * 2), :
            ],
            icvec,
            self.simulation_params.cfl,
        )
        mat_file_path = test_data_dir / f"linear_tx_icmat_t=2.mat"
        test_utils.check_variable(mat_file_path=mat_file_path, var_name=var_name, test_value=icmat)

        icvec = self.wave_transmitter.generate_transmit_pulse(i=2)
        icmat[
            int(self.incoords.shape[0] / 3 * 2) : int(self.incoords.shape[0] + 1), :
        ] = self._focus_coords(
            self.fcen[0],
            self.fcen[1],
            self.incoords[
                int(self.incoords.shape[0] / 3 * 2) : int(self.incoords.shape[0] + 1), :
            ],
            icvec,
            self.simulation_params.cfl,
        )
        mat_file_path = test_data_dir / f"linear_tx_icmat_t=3.mat"
        test_utils.check_variable(mat_file_path=mat_file_path, var_name=var_name, test_value=icmat)

        # 1st
        icmat[
            0 : utils.matlab_round(self.transducer.num_x / 2)
            - utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1 :,
        ] = 0
        icmat[
            utils.matlab_round(self.transducer.num_x / 2)
            + utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1 : self.transducer.num_x,
            :,
        ] = 0
        mat_file_path = test_data_dir / f"linear_tx_icmat_1st.mat"
        test_utils.check_variable(mat_file_path=mat_file_path, var_name=var_name, test_value=icmat)

        # 2nd
        icmat[
            self.transducer.num_x
            + 1
            - 1 : self.transducer.num_x
            + utils.matlab_round(self.transducer.num_x / 2)
            - utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1,
            :,
        ] = 0
        icmat[
            self.transducer.num_x
            + utils.matlab_round(self.transducer.num_x / 2)
            + utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1 : 2 * self.transducer.num_x :,
        ] = 0
        mat_file_path = test_data_dir / f"linear_tx_icmat_2nd.mat"
        test_utils.check_variable(mat_file_path=mat_file_path, var_name=var_name, test_value=icmat)

        # 3rd
        icmat[
            2 * self.transducer.num_x
            + 1
            - 1 : 2 * self.transducer.num_x
            + utils.matlab_round(self.transducer.num_x / 2)
            - utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1,
            :,
        ] = 0
        icmat[
            2 * self.transducer.num_x
            + utils.matlab_round(self.transducer.num_x / 2)
            + utils.matlab_round(self.txducer_aperture * self.beam_spacing / 2)
            - 1 : 3 * self.transducer.num_x :,
        ] = 0
        mat_file_path = test_data_dir / f"linear_tx_icmat_3rd.mat"
        test_utils.check_variable(mat_file_path=mat_file_path, var_name=var_name, test_value=icmat)

        return icmat


def build_full_instance():
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialProperties()

    transducer = L125Transducer(simulation_params, material_properties)
    wave_transmitter = LinearTxWaveTransmitter(
        transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )

    initial_condition = FocusedInitialCondition(
        transducer=transducer,
        wave_transmitter=wave_transmitter,
        simulation_params=simulation_params,
    )
    return initial_condition


def build_test_instance():
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialProperties()

    transducer = L125Transducer(simulation_params, material_properties)
    wave_transmitter = LinearTxWaveTransmitter(
        transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )

    initial_condition = InitialConditionMod(
        transducer=transducer,
        wave_transmitter=wave_transmitter,
        simulation_params=simulation_params,
    )
    return initial_condition


def test_instance():
    initial_condition = build_full_instance()
    assert isinstance(initial_condition, FocusedInitialCondition)


def test_generate_icmat():
    initial_condition = build_test_instance()
    _ = initial_condition.generate_icmat()


def test_focus_coords():
    initial_condition = build_full_instance()
    test_data_dir = test_utils.get_test_data_dir("transducers")

    idy = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "linear_tx_focus_coords_idy_t=1.mat", var_name="idy"
    ).astype(float)[0, 0]
    idz = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "linear_tx_focus_coords_idz_t=1.mat", var_name="idz"
    ).astype(float)[0, 0]
    coords = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "linear_tx_focus_coords_coords_t=1.mat", var_name="coords"
    ).astype(float)
    icvec = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "linear_tx_focus_coords_icvec_t=1.mat", var_name="icvec"
    ).astype(float)[0]
    cfl = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "linear_tx_focus_coords_cfl_t=1.mat", var_name="cfl"
    ).astype(float)[0, 0]

    icmat = initial_condition._focus_coords(idy, idz, coords, icvec, cfl)
    mat_file_path = test_data_dir / f"linear_tx_focus_coords_icmat_t=1.mat"
    test_utils.check_variable(mat_file_path=mat_file_path, var_name="icmat", test_value=icmat)
