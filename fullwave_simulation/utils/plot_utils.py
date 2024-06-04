import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_wave_propagation_animation(
    receivier_whole_map,
    dpi,
    num_plot_image=50,
    export_name="anim.mp4",
    vmin=None,
    vmax=None,
    resize_factor=1,
):
    nt = receivier_whole_map.shape[0]
    skip_every_n_frame = int(nt / num_plot_image)
    plt.close("all")
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    start = 0
    end = None
    animation_list = []
    for i, p_map_i in tqdm(
        enumerate(receivier_whole_map[::skip_every_n_frame, start:end, start:end]),
        total=len(receivier_whole_map[::skip_every_n_frame, start:end, start:end]),
        desc="plotting animation",
    ):
        if resize_factor != 1:
            new_width = int(p_map_i.shape[1] * resize_factor)
            new_height = int(p_map_i.shape[0] * resize_factor)
            p_map_i = cv2.resize(p_map_i, (new_width, new_height))
        image2 = axes.imshow(
            p_map_i,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        animation_list.append([image2])
    animation_data = animation.ArtistAnimation(
        fig, animation_list, interval=150, blit=True, repeat_delay=500
    )
    animation_data.save(export_name, writer="ffmpeg", dpi=dpi)


def plot_wave_propagation_with_map(
    receivier_whole_map,
    c_map,
    rho_map,
    dpi,
    num_plot_image=50,
    export_name="anim.mp4",
    vmin=None,
    vmax=None,
    resize_factor=1,
):
    nt = receivier_whole_map.shape[0]
    skip_every_n_frame = int(nt / num_plot_image)
    plt.close()
    plt.cla()
    plt.clf()
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    start = 0
    end = None
    z_map = c_map * rho_map
    z_map = (z_map - np.min(z_map)) / (np.max(z_map) - np.min(z_map))
    z_map_offset = vmax * 0.8

    animation_list = []
    for i, p_map_i in tqdm(
        enumerate(receivier_whole_map[::skip_every_n_frame, start:end, start:end]),
        total=len(receivier_whole_map[::skip_every_n_frame, start:end, start:end]),
        desc="plotting animation",
    ):
        p_map_i = p_map_i + z_map_offset * (z_map.T)
        if resize_factor != 1:
            new_width = int(p_map_i.shape[1] * resize_factor)
            new_height = int(p_map_i.shape[0] * resize_factor)
            p_map_i = cv2.resize(p_map_i, (new_width, new_height))
        image2 = axes.imshow(
            p_map_i,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        animation_list.append([image2])
    animation_data = animation.ArtistAnimation(
        fig, animation_list, interval=150, blit=True, repeat_delay=500
    )
    animation_data.save(export_name, writer="ffmpeg", dpi=dpi)
