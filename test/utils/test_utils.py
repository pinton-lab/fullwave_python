import os
import random

import numpy as np

from fullwave_simulation.utils import test_utils, utils


def test_seed_everything():
    # test seed_everything. test that the seed is set correctly
    # and that the same random numbers are generated
    seed = 42
    utils.seed_everything(seed)
    random_number_1 = random.random()
    random_number_2 = random.random()
    assert random_number_1 == 0.6394267984578837
    assert random_number_2 == 0.025010755222666936

    utils.seed_everything(seed)
    random_number_1 = random.random()
    random_number_2 = random.random()
    assert random_number_1 == 0.6394267984578837
    assert random_number_2 == 0.025010755222666936

    # test that the seed is set correctly with numpy
    utils.seed_everything(seed)
    random_number_1 = np.random.random()
    random_number_2 = np.random.random()
    assert random_number_1 == 0.3745401188473625
    assert random_number_2 == 0.9507143064099162

    utils.seed_everything(seed)
    random_number_1 = np.random.random()
    random_number_2 = np.random.random()
    assert random_number_1 == 0.3745401188473625
    assert random_number_2 == 0.9507143064099162

    # test that the seed is set correctly with os
    utils.seed_everything(seed)
    random_number_1 = os.environ["PYTHONHASSEED"]
    assert random_number_1 == "42"


def test_normalize_255():
    # test normalize_255
    image = np.array([1, 2, 3, 4, 5])
    normalized_image = utils.normalize_255(image)
    assert np.allclose(normalized_image, np.array([0, 63.75, 127.5, 191.25, 255]))


def test_matlab_round():
    # test matlab_round
    for value in range(10):
        rounded_value = utils.matlab_round(value + 0.5)
        assert rounded_value == value + 1


def test_matlab_gaussian_filter():
    test_data_dir = test_utils.get_test_data_dir("utils")
    cmap_before = test_utils.load_test_variable(
        mat_file_path=test_data_dir / "cmap_before_gaussian_filter.mat", var_name="cmap"
    )

    sigma = 1.5
    cmap_test = utils.matlab_gaussian_filter(cmap_before, sigma)

    test_utils.check_variable(
        mat_file_path=test_data_dir / "cmap_after_gaussian_filter.mat",
        var_name="cmap",
        test_value=cmap_test,
    )
