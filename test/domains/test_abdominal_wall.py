from pathlib import Path

import numpy as np
import scipy

from fullwave_simulation.constants import (
    Constant,
    FocusedSimulationParams,
    FSASimulationParams,
    MaterialProperties,
)
from fullwave_simulation.domains import AbdominalWall
from fullwave_simulation.transducers import C52VTransducer, L125Transducer
from fullwave_simulation.utils import test_utils, utils


class AbdominalWallMod(AbdominalWall):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        crop_depth: float,
        start_depth: float,
        dX: float,
        dY: float,
        transducer,
        abdominal_wall_mat_path: Path,
        material_properties: MaterialProperties,
        simulation_params: FSASimulationParams,
        apply_tissue_deformation: bool = False,
        apply_tissue_compression: bool = True,
        name="abdominal_wall",
        background_domain_properties="water",
        use_smoothing=False,
        skip_i0=True,
        sequence_type="focused",
        use_center_region=False,
        ppw=12,
    ):
        self.crop_depth = crop_depth
        self.start_depth = start_depth
        self.dX = dX
        self.dY = dY
        self.transducer = transducer
        self.abdominal_wall_mat_path = abdominal_wall_mat_path
        self.apply_tissue_deformation = apply_tissue_deformation
        if apply_tissue_deformation:
            self.tranducer_surface = self.transducer.convex_transmitter_map.surface_label.astype(
                int
            )
        self.apply_tissue_compression = apply_tissue_compression
        self.use_smoothing = use_smoothing
        self.skip_i0 = skip_i0
        self.sequence_type = sequence_type
        self.use_center_region = use_center_region
        self.ppw = ppw
        self.background_domain_properties = background_domain_properties

        self.num_x = num_x
        self.num_y = num_y
        self.material_properties = material_properties
        self.simulation_params = simulation_params
        self.name = name

        # self.geometry = self._setup_geometry()

        self.rho_map = np.zeros((self.num_x, self.num_y))
        self.beta_map = np.zeros((self.num_x, self.num_y))
        self.c_map = np.zeros((self.num_x, self.num_y))
        self.a_map = np.zeros((self.num_x, self.num_y))
        # self.rho_map, self.c_map, self.a_map, self.beta_map = self._setup_maps()


class MaterialPropertiesFocused(Constant):
    fat = {"bovera": 9.6, "alpha": 0.48, "ppower": 1.1, "c0": 1478, "rho0": 950}
    fat["beta"] = 1 + fat["bovera"] / 2

    # liver = {"bovera": 7.6, "alpha": 0.5, "ppower": 1.1, "c0": 1570, "rho0": 1064}
    # liver["beta"] = 1 + liver["bovera"] / 2

    muscle = {"bovera": 9, "alpha": 1.09, "ppower": 1.0, "c0": 1547, "rho0": 1050}
    muscle["beta"] = 1 + muscle["bovera"] / 2

    water = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1480, "rho0": 1000}
    water["beta"] = 1 + water["bovera"] / 2

    skin = {"bovera": 8, "alpha": 2.1, "ppower": 1, "c0": 1498, "rho0": 1000}
    skin["beta"] = 1 + skin["bovera"] / 2

    tissue = {"bovera": 9, "alpha": 0.5, "ppower": 1, "c0": 1540, "rho0": 1000}
    tissue["beta"] = 1 + tissue["bovera"] / 2

    connective = {"bovera": 8, "alpha": 1.57, "ppower": 1, "c0": 1613, "rho0": 1120}
    connective["beta"] = 1 + connective["bovera"] / 2

    blood = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1520, "rho0": 1000}
    blood["beta"] = 1 + blood["bovera"] / 2

    lung_fluid = {"bovera": 5, "alpha": 0.005, "ppower": 2.0, "c0": 1440, "rho0": 1000}
    lung_fluid["beta"] = 1 + lung_fluid["bovera"] / 2

    c0 = 1540
    rho0 = 1000
    a0 = 0.5
    beta0 = 0


def build_full_instance_fsa():
    simulation_params = FSASimulationParams()
    material_properties = MaterialProperties()

    transducer = C52VTransducer(simulation_params, material_properties)
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
    dY = lambda_ / ppw
    depth = 0.8e-2

    abdominal_wall_mat_path = Path(
        "fullwave_simulation/domains/data/abdominal_wall/i2365f_etfw1.mat"
    )
    # abdominal_wall_mat_path = Path("fullwave_simulation/domains/data/set_04.mat")
    assert abdominal_wall_mat_path.exists()

    domain = AbdominalWall(
        num_x=num_x,
        num_y=num_y,
        crop_depth=depth,
        abdominal_wall_mat_path=abdominal_wall_mat_path,
        transducer=transducer,
        dY=dY,
        dX=dY,
        start_depth=0,
        simulation_params=simulation_params,
        material_properties=material_properties,
        apply_tissue_deformation=False,
        apply_tissue_compression=True,
        use_smoothing=False,
        skip_i0=True,
        sequence_type="fsa",
    )
    return domain


def build_full_instance_linear_tx(
    apply_tissue_deformation=False,
    apply_tissue_compression=True,
    use_smoothing=False,
    skip_i0=False,
    is_fsa=False,
):
    simulation_params = FocusedSimulationParams()
    material_properties = MaterialPropertiesFocused()

    transducer = L125Transducer(simulation_params, material_properties)

    abdominal_wall = AbdominalWallMod(
        num_x=transducer.num_x,
        num_y=transducer.num_y,
        crop_depth=0.8e-2,
        # start_depth=0.0124,
        start_depth=0.0,
        # representative_length=0.0124,
        transducer=transducer,
        dY=transducer.dY,
        dX=transducer.dX,
        abdominal_wall_mat_path=Path(
            "fullwave_simulation/domains/data/abdominal_wall/i2365f_etfw1.mat"
        ),
        material_properties=material_properties,
        simulation_params=simulation_params,
        apply_tissue_deformation=apply_tissue_deformation,
        apply_tissue_compression=apply_tissue_compression,
        use_smoothing=use_smoothing,
        skip_i0=skip_i0,
        sequence_type="focused",
        background_domain_properties="lung_fluid",
        use_center_region=True,
    )
    return abdominal_wall


def test_full_instance_fsa():
    domain = build_full_instance_fsa()
    assert isinstance(domain, AbdominalWall)


def test_properties_fsa():
    domain = build_full_instance_fsa()
    geometry = domain.geometry
    assert np.all(np.unique(geometry) == np.array([0, 1, 2, 3]))


def test_setup_geometry():
    test_data_dir = test_utils.get_test_data_dir("domains")
    abdominal_wall = build_full_instance_linear_tx(
        apply_tissue_deformation=False,
        apply_tissue_compression=True,
        use_smoothing=False,
        skip_i0=False,
        is_fsa=False,
    )
    geometry = abdominal_wall._setup_geometry()
    test_utils.check_variable(
        mat_file_path=test_data_dir / "abdominal_wall_cropped.mat",
        var_name="mat3",
        test_value=geometry,
    )


def test_setup_geometry_sequantially():
    test_data_dir = test_utils.get_test_data_dir("domains")
    abdominal_wall = build_full_instance_linear_tx(
        apply_tissue_deformation=False,
        apply_tissue_compression=True,
        use_smoothing=False,
        skip_i0=False,
        is_fsa=False,
    )

    mat_data = scipy.io.loadmat(abdominal_wall.abdominal_wall_mat_path)
    abdominal_wall_properties = mat_data["cut"].T.astype(float)

    test_utils.check_variable(
        mat_file_path=test_data_dir / "abdominal_wall_cut.mat",
        var_name="cut",
        test_value=abdominal_wall_properties.T,
    )

    dm = 0.33e-3 / 4

    interpolation_x = dm / abdominal_wall.dX
    interpolation_y = 0.655 * dm / abdominal_wall.dY

    # x_new = utils.matlab_round(abdominal_wall_properties.shape[0] * interpolation_x)
    # y_new = utils.matlab_round(abdominal_wall_properties.shape[1] * interpolation_y)

    abdominal_wall_properties = utils.matlab_interp2easy(
        abdominal_wall_properties,
        interpolation_x=interpolation_x,
        interpolation_y=interpolation_y,
    )

    test_utils.check_variable(
        mat_file_path=test_data_dir / "abdominal_wall_reseized_compressed.mat",
        var_name="mat3",
        test_value=abdominal_wall_properties.T,
    )

    # this is the pixel size in interpd Visual Human slice.
    # important: crop_depth affects the fascia delineation process and depth of lung
    crop_depth_index = utils.matlab_round(abdominal_wall.crop_depth / abdominal_wall.dY) - 1
    # start_depth_index = utils.matlab_round(abdominal_wall.start_depth / abdominal_wall.dY)

    abdominal_wall_properties = abdominal_wall_properties[
        :, crop_depth_index : crop_depth_index + abdominal_wall.num_y
    ]
    test_utils.check_variable(
        mat_file_path=test_data_dir / "abdominal_wall_cropped.mat",
        var_name="mat3",
        test_value=abdominal_wall_properties,
    )


def test_setup_maps_sequantially():
    test_data_dir = test_utils.get_test_data_dir("domains")
    abdominal_wall = build_full_instance_linear_tx(
        apply_tissue_deformation=False,
        apply_tissue_compression=True,
        use_smoothing=True,
        skip_i0=False,
        is_fsa=False,
    )
    abdominal_wall.geometry = abdominal_wall._setup_geometry()

    rho_map = (
        np.ones_like(abdominal_wall.geometry) * abdominal_wall.material_properties.water["rho0"]
    )
    c_map = np.ones_like(abdominal_wall.geometry) * abdominal_wall.material_properties.water["c0"]
    a_map = (
        np.ones_like(abdominal_wall.geometry) * abdominal_wall.material_properties.water["alpha"]
    )
    beta_map = (
        np.ones_like(abdominal_wall.geometry) * abdominal_wall.material_properties.water["beta"]
    )

    for i, tissue_name in enumerate(
        [
            "lung_fluid",
            "connective",
            "muscle",
            "fat",
            "connective",
            "connective",
        ],
    ):
        # if i == 0:
        #     continue
        target_index = np.where(abdominal_wall.geometry == i)
        rho_map[target_index] = getattr(abdominal_wall.material_properties, tissue_name)["rho0"]
        c_map[target_index] = getattr(abdominal_wall.material_properties, tissue_name)["c0"]
        a_map[target_index] = getattr(abdominal_wall.material_properties, tissue_name)["alpha"]

    test_utils.check_variable(
        mat_file_path=test_data_dir / "cmap_before_gaussian_filter.mat",
        var_name="cmap",
        test_value=c_map,
        export_diff_image_when_false=True,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "rhomap_before_gaussian_filter.mat",
        var_name="rhomap",
        test_value=rho_map,
        export_diff_image_when_false=True,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "Amap_before_gaussian_filter.mat",
        var_name="Amap",
        test_value=a_map,
        export_diff_image_when_false=True,
    )

    if abdominal_wall.use_smoothing:
        # use gaussian smoothing to smooth the abdominal wall
        sigma = (5 / 10) ** 2 * abdominal_wall.simulation_params.ppw / 2
        rho_map = utils.matlab_gaussian_filter(rho_map, sigma=sigma)
        c_map = utils.matlab_gaussian_filter(c_map, sigma=sigma)
        a_map = utils.matlab_gaussian_filter(a_map, sigma=sigma)

    test_utils.check_variable(
        mat_file_path=test_data_dir / "cmap_after_gaussian_filter.mat",
        var_name="cmap",
        test_value=c_map,
        export_diff_image_when_false=True,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "rhomap_after_gaussian_filter.mat",
        var_name="rhomap",
        test_value=rho_map,
        export_diff_image_when_false=True,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "Amap_after_gaussian_filter.mat",
        var_name="Amap",
        test_value=a_map,
        export_diff_image_when_false=True,
    )
    # return rho_map, c_map, a_map, beta_map


def test_setup_maps():
    test_data_dir = test_utils.get_test_data_dir("domains")
    abdominal_wall = build_full_instance_linear_tx(
        apply_tissue_deformation=False,
        apply_tissue_compression=True,
        use_smoothing=True,
        skip_i0=False,
        is_fsa=False,
    )
    abdominal_wall.geometry = abdominal_wall._setup_geometry()
    rho_map, c_map, a_map, beta_map = abdominal_wall._setup_maps()
    test_utils.check_variable(
        mat_file_path=test_data_dir / "cmap_after_gaussian_filter.mat",
        var_name="cmap",
        test_value=c_map,
        export_diff_image_when_false=True,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "rhomap_after_gaussian_filter.mat",
        var_name="rhomap",
        test_value=rho_map,
        export_diff_image_when_false=True,
    )
    test_utils.check_variable(
        mat_file_path=test_data_dir / "Amap_after_gaussian_filter.mat",
        var_name="Amap",
        test_value=a_map,
        export_diff_image_when_false=True,
    )
