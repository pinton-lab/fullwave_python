from pathlib import Path

import numpy as np

import fullwave_simulation
from fullwave_simulation.conditions import FSAInitialCondition
from fullwave_simulation.constants import Constant
from fullwave_simulation.domains import (
    AbdominalWall,
    Background,
    DomainOrganizer,
    PhatomLateral,
    Scatterer,
    WaterGel,
)
from fullwave_simulation.solvers import FullwaveSolver
from fullwave_simulation.transducers import (
    C52VTransducer,
    ConvexTxWaveTransmitter,
    SignalReceiver,
)
from fullwave_simulation.utils import MapViewer


class FSASimulationParams(Constant):
    # 64 sequential txrx events
    nevents = 128

    # element_pitch
    element_pitch = 0.000508  # [m]

    # Basic variables / parameters
    freq_div = 1
    c0 = 1540  # [m/s]

    # frequency [MHz]
    # not realistic radio frequency.
    f0 = 3.7e6 / freq_div  # [Hz]

    omega0 = 2 * np.pi * f0
    lambda_ = c0 / f0

    # Simulation grid varables
    # width of simulation field (m). lateral dimension.
    wX = 6e-2  # [m]
    # depth of simulation field (m)
    wY = 9e-2  # [m]

    # duration of simulation (s).
    # the time how much you want to simulate the propagation (sec)
    dur = wY * 2.3 / c0  # [sec]

    # Courant-Friedrichs-Levi condition
    cfl = 0.4

    # pressure in Pa. 100kpa.
    # 2.5Mpa is normal but not in this simulation.
    p0 = 1e5  # [Pa]

    # emission parameters
    num_cycles = 2
    # exponential drop-off of envelope
    drop_off = 1

    # --- aliases ---
    width = wX
    depth = wY
    modT = 7
    ncycles = num_cycles

    # define fnumber
    fnumber = 1.5
    # fnumber = focal_depth / width

    # whether the transmit sequence is Fully synthetic aperture (FSA)
    is_fsa = True
    # is_fsa = False        self.bw = self.simulation_params.tx_bw
    tx_bw = 0.7


class MaterialProperties(Constant):
    fat = {
        "b_over_a": 9.6,
        "alpha": 0.48,
        "ppower": 1.1,
        "c0": 1478,
        "rho0": 950,
    }
    fat["beta"] = 1 + fat["b_over_a"] / 2

    liver = {
        "b_over_a": 7.6,
        "alpha": 0.5,
        "ppower": 1.1,
        "c0": 1570,
        "rho0": 1064,
    }
    liver["beta"] = 1 + liver["b_over_a"] / 2

    muscle = {
        "b_over_a": 9,
        "alpha": 1.09,
        "ppower": 1.0,
        "c0": 1547,
        "rho0": 1050,
    }
    muscle["beta"] = 1 + muscle["b_over_a"] / 2

    water = {
        "b_over_a": 5,
        "alpha": 0.005,
        "ppower": 2.0,
        "c0": 1480,
        "rho0": 1000,
    }
    water["beta"] = 1 + water["b_over_a"] / 2

    skin = {
        "b_over_a": 8,
        "alpha": 2.1,
        "ppower": 1,
        "c0": 1498,
        "rho0": 1000,
    }
    skin["beta"] = 1 + skin["b_over_a"] / 2

    tissue = {
        "b_over_a": 9,
        "alpha": 0.5,
        "ppower": 1,
        "c0": 1540,
        "rho0": 1000,
    }
    tissue["beta"] = 1 + tissue["b_over_a"] / 2

    connective = {
        "b_over_a": 8,
        "alpha": 1.57,
        "ppower": 1,
        "c0": 1613,
        "rho0": 1120,
    }
    connective["beta"] = 1 + connective["b_over_a"] / 2

    blood = {
        "b_over_a": 5,
        "alpha": 0.005,
        "ppower": 2.0,
        "c0": 1520,
        "rho0": 1000,
    }
    blood["beta"] = 1 + blood["b_over_a"] / 2

    lung_fluid = {
        "b_over_a": 5,
        "alpha": 0.005,
        "ppower": 2.0,
        "c0": 1440,
        "rho0": 1000,
    }
    lung_fluid["beta"] = 1 + lung_fluid["b_over_a"] / 2

    transducer = connective

    c0 = 1540
    rho0 = 1000
    a0 = 0.5
    beta0 = 0


def main():
    # Define your work directory and make the directory.
    home_dir = Path(fullwave_simulation.__file__).parent.parent
    work_dir = home_dir / "outputs" / "exp_dir_20240603_test"
    work_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    # Set the parameters with fullwave_simulation.constants classes.
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    # Define the transducer properties using class in `fullwave_simulation.transducers`.
    # C52V is available at the moment. More options will be available such as a L7-4 and L12-5 (linear transducer).
    c52v_transducer = C52VTransducer(
        simulation_params=simulation_params,
        material_properties=material_properties,
        rho0=material_properties.transducer["rho0"],
        c0=material_properties.transducer["c0"],
        a0=material_properties.transducer["alpha"],
        beta0=material_properties.transducer["beta"],
        element_pitch=FSASimulationParams.element_pitch,
    )

    # Define the simulation domains using fullwave_simulation.domains classes.
    # Each domain has its own material properties like density, sound speed, attenuation, etc.
    # If you need to make a new simulational maps or domains such as abdmonial wall, lung, or liver,
    # you will write a class refer to these classes.
    background_domain_properties = "liver"
    map_viewer = MapViewer(save_dir=work_dir / "input_maps")

    # In this example, background with scatter, abdominal wall, and phantom were defined.
    # First, download the abdominal wall data and put them to `fullwave_simulation/domains/data`
    # https://drive.google.com/file/d/1KMSlqcgXSzd9NGU2fauO9OJ6s8PPrA5P/view?usp=sharing
    background = Background(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        material_properties=material_properties,
        simulation_params=simulation_params,
        background_domain_properties=background_domain_properties,
    )
    scatterer = Scatterer(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        material_properties=material_properties,
        simulation_params=simulation_params,
        transducer=c52v_transducer,
        num_scatter=18,
    )
    csr = 0.035
    background.rho_map = background.rho_map - scatterer.rho_map * csr

    phantom = PhatomLateral(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        material_properties=material_properties,
        simulation_params=simulation_params,
        dX=c52v_transducer.dX,
        dY=c52v_transducer.dY,
        base_circle_depth_in_meter=0.04,
        lat_phantom_in_meter=simulation_params.wX,
        depth_phantom_in_meter=simulation_params.wY,
        background_domain_properties=background_domain_properties,
    )
    abdominal_wall = AbdominalWall(
        num_x=c52v_transducer.num_x,
        num_y=c52v_transducer.num_y,
        crop_depth=0.8e-2,
        start_depth=0.0,
        dY=c52v_transducer.dY,
        dX=c52v_transducer.dX,
        transducer=c52v_transducer,
        abdominal_wall_mat_path=Path(
            "fullwave_simulation/domains/data/abdominal_wall/i2365f_etfw1.mat"
        ),
        material_properties=material_properties,
        simulation_params=simulation_params,
        apply_tissue_deformation=True,
        apply_tissue_compression=True,
        use_smoothing=True,
        skip_i0=False,
        use_center_region=True,
        background_domain_properties=background_domain_properties,
        ppw=c52v_transducer.ppw,
        sequence_type="fsa",
    )

    # Next, register each domain classes into DomainOrganizer and construct a integrated domain.
    # The order of the domains is important.
    # The domain map will be constructed in a bottom-up fashion with DomainOrganizer like a sticker using the registered domains.
    domain_organizer = DomainOrganizer(
        material_properties=material_properties,
        background_domain_properties=background_domain_properties,
        ignore_non_linearity=True,
    )
    domain_organizer.register_domains(
        [
            background,
            phantom,
            abdominal_wall,
            c52v_transducer.convex_transmitter_map,
        ],
    )
    domain_organizer.construct_domain()

    # you can view the constructed domain maps using MapViewer.
    for map_type in [
        "rho_map",
        "beta_map",
        "c_map",
        "a_map",
        "geometry",
    ]:
        map_viewer.view_map(
            # np.flip(domain_organizer.constructed_domain_dict[map_type].T, 1),
            map_data=domain_organizer.constructed_domain_dict[map_type].T,
            title=map_type,
            save_name_base=map_type,
            extent=[
                -simulation_params.wX / 2 * 1e3,
                simulation_params.wX / 2 * 1e3,
                simulation_params.wY * 1e3,
                0,
            ],
        )

    # Now, define the wave transmitter and signal receiver.
    # WaveTransmitter is used to calculate the transmission pulse.
    # SignalReceiver does not have an effect at the moment.
    wave_transmitter = ConvexTxWaveTransmitter(
        transducer=c52v_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
        is_fsa=simulation_params.is_fsa,
    )
    # signal_receiver is a dummy for now.
    signal_receiver = SignalReceiver(
        transducer=c52v_transducer,
        simulation_params=simulation_params,
        material_properties=material_properties,
    )

    # Define the initial condition.
    # InitialCondition class is used to generate the icmat,
    # which is the initial pressure in time space,
    # for each event based on the transmission pulse (icvec).
    # icvec will be generated by the WaveTransmitter.
    initial_condition = FSAInitialCondition(
        # is_fsa=simulation_params.is_fsa,
        transducer=c52v_transducer,
        wave_transmitter=wave_transmitter,
    )

    # Finally, pass the above defined parameters to the solver and run the simulation.
    # genout_list contains numpy array version of the genout,
    # which is a Fullwave2's output file.
    # Each outputs will be exported in the work directory defined in a first step.
    fw_solver = FullwaveSolver(
        work_dir=work_dir,
        #
        simulation_params=simulation_params,
        #
        domain_organizer=domain_organizer,
        transducer=c52v_transducer,
        wave_transmitter=wave_transmitter,
        signal_receiver=signal_receiver,
        #
        initial_condition=initial_condition,
        on_memory=False,
    )
    genout_list = fw_solver.run()
    print()


if __name__ == "__main__":
    main()
