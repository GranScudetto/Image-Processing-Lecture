from pathlib import Path
from typing import Tuple

import click
import matplotlib.pyplot as plt
import numpy as np

RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2


def debayer_rgb_image_with_nearest_neighbor_per_indexing(
        bayer_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Variant of the upper function with rgb images as input
    new_image = np.zeros((bayer_image.shape[0], bayer_image.shape[1], 3))
    bayer_pattern = np.copy(new_image)

    # Again ignoring the border
    # reading the corresponding RGB values from their position in the bayer pattern
    # Position = Red Pixel (R1)
    new_image[1:-1, 1:-1, :][1::2, 1::2,
                             RED_CHANNEL] = bayer_image[1:-1, 1:-1, :][1::2, 1::2, RED_CHANNEL]
    new_image[1:-1, 1:-1, :][1::2, 1::2, GREEN_CHANNEL] = (
        bayer_image[:-2, 1:-1, :][1::2, 1::2, GREEN_CHANNEL] +  # -1 0
        bayer_image[2:, 1:-1, :][1::2, 1::2, GREEN_CHANNEL] +  # +1 0
        bayer_image[1:-1, :-2, :][1::2, 1::2, GREEN_CHANNEL] +  # 0 -1
        bayer_image[1:-1, 2:, :][1::2, 1::2, GREEN_CHANNEL]) / 4  # 0 +1
    new_image[1:-1, 1:-1, :][1::2, 1::2, BLUE_CHANNEL] = (
        bayer_image[:-2, :-2, :][1::2, 1::2, BLUE_CHANNEL] +  # -1 -1
        bayer_image[:-2, 2:, :][1::2, 1::2, BLUE_CHANNEL] +  # -1 +1
        bayer_image[2:, 2:, :][1::2, 1::2, BLUE_CHANNEL] +  # +1 +1
        bayer_image[2:, :-2, :][1::2, 1::2, BLUE_CHANNEL]) / 4  # +1 -1

    # Position = Green Pixel (G1)
    new_image[1:-1, 1:-1, :][
        ::2, 1::2, GREEN_CHANNEL] = bayer_image[1:-1, 1:-1, :][::2, 1::2, GREEN_CHANNEL]
    new_image[1:-1, 1:-1, :][::2, 1::2, RED_CHANNEL] = (
        bayer_image[2:, 1:-1, :][::2, 1::2, RED_CHANNEL] +  # +1 0
        bayer_image[:-2, 1:-1, :][::2, 1::2, RED_CHANNEL]) / 2  # -1 0
    new_image[1:-1, 1:-1, :][::2, 1::2, BLUE_CHANNEL] = (
        bayer_image[1:-1, 2:, :][::2, 1::2, BLUE_CHANNEL] +  # 0 +1
        bayer_image[1:-1, :-2, :][::2, 1::2, BLUE_CHANNEL]) / 2  # 0 -1

    # Position = Green Pixel (G2)
    new_image[1:-1, 1:-1, :][
        1::2, ::2, GREEN_CHANNEL] = bayer_image[1:-1, 1:-1, :][1::2, ::2, GREEN_CHANNEL]
    new_image[1:-1, 1:-1, :][1::2, ::2, RED_CHANNEL] = (
        bayer_image[1:-1, 2:, :][1::2, ::2, RED_CHANNEL] +  # 0 +1
        bayer_image[1:-1, :-2, :][1::2, ::2, RED_CHANNEL]) / 2  # 0 -1
    new_image[1:-1, 1:-1, :][1::2, ::2, BLUE_CHANNEL] = (
        bayer_image[2:, 1:-1, :][1::2, ::2, BLUE_CHANNEL] +  # +1 0
        bayer_image[:-2, 1:-1, :][1::2, ::2, BLUE_CHANNEL]) / 2  # -1 0

    # Position = Blue pixel (B1)
    new_image[1:-1, 1:-1, :][
        ::2, ::2, BLUE_CHANNEL] = bayer_image[1:-1, 1:-1, :][::2, ::2, BLUE_CHANNEL]
    new_image[1:-1, 1:-1, :][::2, ::2, GREEN_CHANNEL] = (
        bayer_image[:-2, 1:-1, :][::2, ::2, GREEN_CHANNEL] +  # -1 0
        bayer_image[2:, 1:-1, :][::2, ::2, GREEN_CHANNEL] +  # +1 0
        bayer_image[1:-1, :-2, :][::2, ::2, GREEN_CHANNEL] +  # 0 -1
        bayer_image[1:-1, 2:, :][::2, ::2, GREEN_CHANNEL]) / 4  # 0 +1
    new_image[1:-1, 1:-1, :][::2, ::2, RED_CHANNEL] = (
        bayer_image[:-2, :-2, :][::2, ::2, RED_CHANNEL] +  # -1 -1
        bayer_image[:-2, 2:, :][::2, ::2, RED_CHANNEL] +  # -1 + 1
        bayer_image[2:, 2:, :][::2, ::2, RED_CHANNEL] +  # +1 +1
        bayer_image[2:, :-2, :][::2, ::2, RED_CHANNEL]) / 4  # +1 -1

    # store bayer pattern for reference (due to the border ::2 -> 1::2 and vice versa
    # R = ::2, ::2, # G1 = 1::2, ::2, # G2 = ::2, 1::2, # B = 1::2, 1::2
    bayer_pattern[1:-1, 1:-1, :][1::2, 1::2, RED_CHANNEL] = 1
    bayer_pattern[1:-1, 1:-1, :][::2, 1::2, GREEN_CHANNEL] = 1
    bayer_pattern[1:-1, 1:-1, :][1::2, ::2, GREEN_CHANNEL] = 1
    bayer_pattern[1:-1, 1:-1, :][::2, ::2, BLUE_CHANNEL] = 1

    return new_image, bayer_pattern


def check_for_manipulation(suspicious_image: np.ndarray) -> None:
    image_debayered, _ = debayer_rgb_image_with_nearest_neighbor_per_indexing(
        suspicious_image)
    # check if the "bayer" correlation still holds true
    is_close = np.isclose(
        suspicious_image[:, :, :3], image_debayered[:, :, :], atol=5e-3)

    # set True to False and False to True to reflect the manipulation instead of that the values are close
    manipulated = np.invert(is_close)
    # Sum along the R, G, B Channels
    coordinates = np.sum(manipulated, axis=2)
    # highlight each image position wether it is R, G or B with a 1
    positions = np.zeros(
        (suspicious_image.shape[0], suspicious_image.shape[1]))
    positions[np.where(coordinates > 0)] = 1

    fig, axes = plt.subplots(nrows=2, ncols=3, sharey='all', sharex='all')
    axes[0, 0].imshow(suspicious_image)
    axes[0, 0].set_title('Original image')
    axes[0, 1].imshow(manipulated.astype(int) * 255)
    axes[0, 1].set_title('Suspicious Areas')
    axes[0, 2].imshow(positions.astype(int), cmap='coolwarm')
    axes[1, 0].imshow(manipulated[:, :, 0].astype(int) * 255, cmap='coolwarm')
    axes[1, 0].set_title('Red Channel')
    axes[1, 1].imshow(manipulated[:, :, 0].astype(int) * 255, cmap='coolwarm')
    axes[1, 1].set_title('Green Channel')
    axes[1, 2].imshow(manipulated[:, :, 0].astype(int) * 255, cmap='coolwarm')
    axes[1, 2].set_title('Blue Channel')
    plt.show()


@click.command()
@click.argument('filename', type=click.Path(exists=True))
def main(filename: Path) -> None:
    image = plt.imread(filename)
    check_for_manipulation(image)


if __name__ == '__main__':
    main()

