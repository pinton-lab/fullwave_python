# Fullwave 2 simulation python wrapper

Python wrapper for the Fullwave 2 simulation.

[Fullwave 2](http://arxiv.org/abs/2106.11476
) is a high-fidelity, finite-difference time-domain (FDTD) ultrasound wave propagation simulation.

It includes a comprehensive set of wave propagation effects, including reverberation, aberration, and nonlinear propagation, that occur in heterogeneous tissue.

This wrapper is designed to make the simulation easy to use from the Python interface.

For more information about the simulation method, please see

- Pinton, G. (2021). A fullwave model of the nonlinear wave equation with multiple relaxations and relaxing perfectly matched layers for high-order numerical finite-difference solutions. In arXiv [physics.med-ph]. arXiv. http://arxiv.org/abs/2106.11476

## requirements

### dev requirements

This repository uses `pre-commit` to standarize the code-style using auto formatter ([black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/))

- base develop environment
  - text editor: [vscode](https://code.visualstudio.com/) prefered.
  - [homebrew](https://brew.sh/)
    - for pyenv, pre-commit
    - follow [here](https://brew.sh/) to install.
    - If you don't have the sudo privilege, follow [this installation](https://docs.brew.sh/Installation#alternative-installs).
  - [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
  - [pre-commit](https://pre-commit.com/)
- python specific dependencies
  - [pyenv](https://github.com/pyenv/pyenv#getting-pyenv)
    - `make install-pyenv`
  - [poetry](https://python-poetry.org/docs/#installation)
    - `make install-poetry`
  - `make install-precommit`

## installation

Install a certain python version using pyenv, if you don't have the required python.

```sh
make install-python
```

then, type below to install the entire package.

```sh
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
make install
```

## usage

The code is modular. First, the individual classes are defined to set up the simulation parameters, domain, transducer properties, initial condition. Then passed them to the `FullwaveSolver` class to construct the simulation.

For more information, please refer th the [Examples](https://github.com/gfpinton/fullwave_python/tree/main/examples).

we provided

### transabdominal ultrasound simulation with plane wave sequence using linear transducer (L12-5)

- [ex_plane_wave_linear_transducer_transabdominal_imaging](https://github.com/gfpinton/fullwave_python/blob/main/examples/ex_plane_wave_linear_transducer_transabdominal_imaging.py)

![plane wave](/figs/plane_wave_anim.gif "Title")

### transabdominal ultrasound simulation with focused sequence using linear transducer (L12-5)

- [ex_focused_linear_transducer_transabdominal_imaging](https://github.com/gfpinton/fullwave_python/blob/main/examples/ex_focused_linear_transducer_transabdominal_imaging.py)

![focused sequence](/figs/focus_anim.gif "Title")

### transabdominal ultrasound simulation with Full synthetic aperture using convex transducer (C5-2V)

- [ex_fsa_convex_transducer_transabdominal_imaging](https://github.com/gfpinton/fullwave_python/blob/main/examples/ex_fsa_convex_transducer_transabdominal_imaging.py)

![FSA](/figs/fsa_anim.gif "Title")

### step-by-step setup for an FSA experiment with phantom image

First, download the abdominal wall data and put them to `fullwave_simulation/domains/data`

- [Abdominal wall data link](https://drive.google.com/file/d/1KMSlqcgXSzd9NGU2fauO9OJ6s8PPrA5P/view?usp=sharing)

Define your work directory and make the directory.

```py
from pathlib import Path
home_dir = Path(fullwave_simulation.__file__).parent.parent
work_dir = home_dir / "exp_dir_20231012_seed42"
work_dir.mkdir(exist_ok=True, parents=True)
```

Set the parameters with `fullwave_simulation.constants` classes.

```py
# define simulation parameters and material parameters
from fullwave_simulation.constants import FSASimulationParams, MaterialProperties

simulation_params = FSASimulationParams()
material_properties = MaterialProperties()
```

Define the transducer properties using class in `fullwave_simulation.transducers`. C52V is available at the moment. More options will be available such as a L7-4 and L12-5 (linear transducer).

```py
# define the transducer properties
from fullwave_simulation.transducers import C52VTransducer

c52v_transducer = C52VTransducer(simulation_params, material_properties)
```

Define the simulation domains using `fullwave_simulation.domains` classes.
Each domain has its own material properties like density, sound speed, attenuation, etc.
If you need to make a new simulational maps or domains sucha as abdmonial wall, lung, or liver, you will write a class refer to these classes.

In this example, background with scatter, water gel, and phantom were defined.

```py
from fullwave_simulation.domains import (
    Background,
    Phantom,
    Scatter,
    WaterGel,
)

background = Background(
    c52v_transducer.num_x,
    c52v_transducer.num_y,
    material_properties,
    simulation_params,
)
scatter = Scatter(
    num_x=c52v_transducer.num_x,
    num_y=c52v_transducer.num_y,
    material_properties=material_properties,
    simulation_params=simulation_params,
    transducer=c52v_transducer,
)
csr = 0.035
background.rho_map = background.rho_map - scatter.rho_map * csr
water_gel = WaterGel(
    num_x=c52v_transducer.num_x,
    num_y=c52v_transducer.num_y,
    depth=0.0124,
    dY=c52v_transducer.dY,
    material_properties=material_properties,
    simulation_params=simulation_params,
)
phantom = Phantom(
    c52v_transducer.num_x,
    c52v_transducer.num_y,
    material_properties,
    simulation_params,
    c52v_transducer.dX,
)
```

Next, register each domain classes into `DomainOrganizer` and construct a integrated domain.
The order of the domains is important. The domain map will be constructed in a bottom-up fashion with `DomainOrganizer` lika a sticker using the registered domains.

```py
from fullwave_simulation.domains import (
    DomainOrganizer,
)

domain_organizer = DomainOrganizer()
domain_organizer.register_domains(
    [
        background,
        water_gel,
        phantom,
        c52v_transducer.convex_transmitter_map,
    ],
)
domain_organizer.construct_domain()
```

Now, define the wave transmitter and signal receiver. `WaveTransmitter` is used to calculate the transmission pulse. `SignalReceiver` does not have an effect at the moment.

```py
from fullwave_simulation.transducers import (
    SignalReceiver,
    WaveTransmitter,
)

wave_transmitter = WaveTransmitter(
    c52v_transducer,
    simulation_params=simulation_params,
    material_properties=material_properties,
    is_fsa=simulation_params.is_fsa,
)
signal_receiver = SignalReceiver(
    c52v_transducer,
    simulation_params=simulation_params,
    material_properties=material_properties,
)
```

Define the initial condition. `InitialCondition` class is used to generate the `icmat`, which is the initial pressure in time space, for each event based on the transmission pulse (`icvec`). icvec will be generated by the `WaveTransmitter`.

```py
from fullwave_simulation.conditions import InitialCondition

initial_condition = InitialCondition(
    is_fsa=simulation_params.is_fsa,
    transducer=c52v_transducer,
    wave_transmitter=wave_transmitter,
)
```

Finally, pass the above defined parameters to the solver and run the simulation. genout_list contains numpy array version of the `genout`, which is a Fullwave2's output file. Each outputs will be exported in the work directory defined in a first step.

```py
from fullwave_simulation.solvers import FullwaveSolver

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
```

## advanced usage

If you want to modify the simulation or set up a new experiment, you only need to inherit the class and modify it without changing the original source code. This is benefitial regarding the reproducibility of the experiment. I will write the experiment setup later.
