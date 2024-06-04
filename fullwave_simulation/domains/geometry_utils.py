import numpy as np


def make_circle_idx(dims, cen, rad):
    x, y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing="ij")
    dist = np.sqrt(np.round((x - cen[0]) + 1e-9) ** 2 + np.round((y - cen[1]) + 1e-9) ** 2)
    return dist <= rad


def map_to_coordinates(map_data):
    idx, idy = np.where(map_data != 0)
    if idx.shape[0] == 0 or idy.shape[0] == 0:
        return np.array([[], []]).T
    coords = np.array([idx, idy])
    unique_num_list = np.unique(coords[1])
    unique_num_list.sort()
    out_list = []
    for value in unique_num_list:
        out_list.append(np.sort(coords[:, coords[1] == value]))
    coords = np.concatenate(out_list, axis=1)
    return coords


def map_to_coords_with_sort(map_data):
    coords = map_to_coordinates(map_data)
    return coords[:, np.argsort(coords[0], kind="mergesort")].T


def map_to_coordinates_matlab(map_data):
    idx, idy = np.where(map_data.T != 0)
    return np.array([idy, idx]).T
