import difflib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy


def debug_image_plot(array, export_path):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plt.title(export_path)
    image = axes[0].imshow(array, interpolation="nearest")
    axes[0].tick_params(axis="x", labelsize=1)
    axes[0].tick_params(axis="y", labelsize=1)

    # axes[0].axis("off")
    axes[0].set_xlabel(f"x {array.shape[1]}")
    axes[0].set_ylabel(f"y {array.shape[0]}")

    step = 100
    axes[0].set_xticks(np.arange(-step, array.shape[1] + 2 * step, step))
    axes[0].set_yticks(np.arange(-step, array.shape[0] + 2 * step, step))
    cbar = plt.colorbar(image)
    cbar.set_label("image value")
    fontsize = 10
    axes[1].text(0.2, 0.8, f"min: {array.min():.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.7, f"max: {array.max():.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.6, f"mean: {array.mean():.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.5, f"median: {np.median(array):.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.4, f"sum: {np.sum(array):.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.3, f"non-zero num: {np.sum(array!=0):.4e}", fontsize=fontsize)
    axes[1].text(
        0.2, 0.2, f"shape (x, y) = ({array.shape[1]}, {array.shape[0]})", fontsize=fontsize
    )
    axes[1].axis("off")

    plt.savefig(export_path, dpi=500)
    plt.cla()
    plt.clf()


def debug_sequence_plot(array, export_path):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plt.title(export_path)
    image = axes[0].plot(array)
    axes[0].tick_params(axis="x", labelsize=1)
    axes[0].tick_params(axis="y", labelsize=1)

    # axes[0].axis("off")
    # axes[0].set_xlabel(f"x {array.shape[1]}")
    # axes[0].set_ylabel(f"y {array.shape[0]}")

    # step = 100
    # axes[0].set_xticks(np.arange(-step, array.shape[1] + 2 * step, step))
    # axes[0].set_yticks(np.arange(-step, array.shape[0] + 2 * step, step))
    # cbar = plt.colorbar(image)
    # cbar.set_label("image value")
    fontsize = 10
    axes[1].text(0.2, 0.8, f"min: {array.min():.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.7, f"max: {array.max():.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.6, f"mean: {array.mean():.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.5, f"median: {np.median(array):.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.4, f"sum: {np.sum(array):.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.3, f"non-zero num: {np.sum(array!=0):.4e}", fontsize=fontsize)
    axes[1].text(0.2, 0.2, f"shape = ({array.shape[0]}", fontsize=fontsize)
    axes[1].axis("off")

    plt.savefig(export_path, dpi=500)
    plt.cla()
    plt.clf()


def get_test_data_dir(test_type):
    test_data_dir = Path(f"test/{test_type}/test_data")
    return test_data_dir


def load_test_variable(mat_file_path, var_name):
    if not mat_file_path.exists():
        raise ValueError(f"mat_file_path {mat_file_path} does not exist")
    test_variable_dict = scipy.io.loadmat(mat_file_path)
    try:
        test_variable = test_variable_dict[var_name]
    except KeyError:
        raise KeyError(f"{mat_file_path}: {var_name} does not exist")
    return test_variable


def load_dat_data(dat_file_path, dtype=np.float32):
    if not dat_file_path.exists():
        raise ValueError(f"dat_file_path {dat_file_path} does not exist")
    return np.fromfile(dat_file_path, dtype=dtype)


def load_text_dat(test_file_path):
    if not test_file_path.exists():
        raise ValueError(f"test_file_path {test_file_path} does not exist")
    with open(test_file_path, "r") as file:
        text_value = file.readlines()
    return text_value


def check_variable(
    mat_file_path, var_name, test_value, rtol=1e-06, export_diff_image_when_false=False
):
    val_ground_truth = load_test_variable(mat_file_path=mat_file_path, var_name=var_name)

    if isinstance(test_value, np.ndarray) and isinstance(val_ground_truth, np.ndarray):
        if test_value.ndim == 1 and val_ground_truth.ndim == 2:
            if test_value.shape[0] == val_ground_truth.shape[1]:
                test_value = np.expand_dims(test_value, 0)
            elif test_value.shape[0] == val_ground_truth.shape[0]:
                test_value = np.expand_dims(test_value, 1)
            else:
                raise ValueError(
                    f"{mat_file_path}: test_value {var_name} and ground_truth are not the same shape"
                )

        if not test_value.shape == val_ground_truth.shape:
            raise ValueError(
                f"{mat_file_path}: test_value {var_name} and ground_truth are not the same shape"
            )
    if not np.allclose(test_value, val_ground_truth, rtol=rtol):
        # index_mismatch = np.where(test_value != val_ground_truth)
        if export_diff_image_when_false:
            debug_image_plot(
                test_value.astype(float), f"{mat_file_path.stem}_{var_name}_test_value.png"
            )
            debug_image_plot(
                val_ground_truth.astype(float),
                f"{mat_file_path.stem}_{var_name}_val_ground_truth.png",
            )
            debug_image_plot(
                val_ground_truth.astype(float) - test_value.astype(float),
                f"{mat_file_path.stem}_{var_name}_diff.png",
            )
            diff_index = np.where(~np.isclose(val_ground_truth, test_value))

        raise ValueError(
            f"{mat_file_path}: test_value {var_name} and ground_truth {val_ground_truth} are not close"
        )


def check_variable_index(mat_file_path, var_name, test_value, rtol=1e-06):
    val_ground_truth = load_test_variable(mat_file_path=mat_file_path, var_name=var_name).astype(
        int
    )
    if not test_value.shape == val_ground_truth.shape:
        raise ValueError(f"test_value {var_name} and ground_truth are not the same shape")
    if not np.allclose(np.sort(test_value), np.sort(val_ground_truth), rtol=rtol):
        # index_mismatch = np.where(test_value != val_ground_truth)
        raise ValueError(
            f"test_value {var_name} and ground_truth {val_ground_truth} are not close"
        )


def load_and_check_dat_data(
    dat_file_path,
    test_dat_file_path,
    dtype=np.float32,
    rtol=1e-06,
    array_shape=None,
    export_diff_image_when_false=False,
):
    test_value = load_dat_data(dat_file_path, dtype=dtype)
    ground_truth_value = load_dat_data(test_dat_file_path, dtype=dtype)
    if not np.allclose(test_value, ground_truth_value, rtol=rtol):
        if array_shape and export_diff_image_when_false:
            test_value_reshaped = test_value.reshape(array_shape).astype(float)
            ground_truth_value_reshaped = ground_truth_value.reshape(array_shape).astype(float)
            debug_image_plot(test_value_reshaped, f"{dat_file_path.stem}_test_value.png")
            debug_image_plot(
                ground_truth_value_reshaped,
                f"{dat_file_path.stem}_val_ground_truth.png",
            )
            debug_image_plot(
                ground_truth_value_reshaped - test_value_reshaped,
                f"{dat_file_path.stem}_diff.png",
            )
            diff_index = np.where(~np.isclose(ground_truth_value_reshaped, test_value_reshaped))
            # for i in range(diff_index[0].shape[0]):
            #     print(f"i={i} x:{diff_index[0][i]} y: {diff_index[1][i]}")
            #     print(f"test: {test_value_reshaped[diff_index[0][i], diff_index[1][i]]}")
            #     print(f"gt: {ground_truth_value_reshaped[diff_index[0][i], diff_index[1][i]]}")
        raise ValueError(
            f"{dat_file_path}: test_value {test_value} and ground_truth {ground_truth_value} are not close"
        )


def load_and_check_text_data(test_file_path, ground_truth_text_file_path):
    test_value = load_text_dat(test_file_path)
    ground_truth_value = load_text_dat(ground_truth_text_file_path)
    diff_list = []
    for line in difflib.unified_diff(
        test_value,
        ground_truth_value,
        fromfile=str(ground_truth_text_file_path),
        tofile=str(test_file_path),
        lineterm="",
    ):
        diff_list.append(line)
    if len(diff_list) > 0:
        raise ValueError(f"text are not same: {diff_list}")
