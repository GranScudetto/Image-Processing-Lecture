import time
from pathlib import Path
from typing import Tuple

import click
import matplotlib.pyplot as plt
import numpy as np

RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2


def debayer_image_with_nearest_neighbor_iterative(
        bayer_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    new_image = np.zeros((bayer_image.shape[0], bayer_image.shape[1], 3))
    bayer_pattern = np.copy(new_image)
    image_rows, image_columns = bayer_image.shape

    # ignore image border in first iteration (no full neighborhood)
    # iterate each pixel
    for row in range(1, image_rows - 1):  # iterate rows
        for column in range(1, image_columns - 1):  # iterate columns
            #       columns ->
            #  rows |   R   |   G   |
            #   |   |-------|-------|
            #  \|/  |   G   |   B   |

            # odd row (starts with index 0 (0, 2, 4, 6, ...)) => either R or G
            if row % 2 == 0:
                # |   R   |   G   |
                # |-------|-------|
                # |       |       |

                if column % 2 == 1:  # even column (1, 3, ...)
                    # |    |    |    |   read Green Pixel Value
                    # |    |  G |    |
                    # |    |    |    |
                    new_image[row, column,
                              GREEN_CHANNEL] = bayer_image[row, column]
                    bayer_pattern[row, column, GREEN_CHANNEL] = 1

                    # |    |     |    |   interpolate red channel value
                    # | X  |  G  |  X |
                    # |    |     |    |
                    new_image[row, column, RED_CHANNEL] = (int(
                        bayer_image[row, column - 1]) + int(
                        bayer_image[row, column + 1])) / 2

                    # |    |  X  |    |   interpolate blue channel value
                    # |    |  G  |    |
                    # |    |  X  |    |
                    new_image[row, column, BLUE_CHANNEL] = (int(
                        bayer_image[row - 1, column]) + int(
                        bayer_image[row + 1, column])) / 2

                else:
                    # |    |    |    |   read Red Pixel Value
                    # |    |  R |    |
                    # |    |    |    |
                    new_image[row, column,
                              RED_CHANNEL] = bayer_image[row, column]
                    bayer_pattern[row, column, RED_CHANNEL] = 1

                    # |    |  X  |    |   interpolate green channel value
                    # | X  |  R  |  X |
                    # |    |  X  |    |
                    new_image[row, column, GREEN_CHANNEL] = (int(
                        bayer_image[row - 1, column]) + int(
                        bayer_image[row + 1, column]) + int(
                        bayer_image[row, column + 1]) + int(
                        bayer_image[row, column - 1])) / 4

                    # | X  |     |    |   interpolate blue channel value
                    # |    |  R  |    |
                    # |    |     |  X |
                    new_image[row, column, BLUE_CHANNEL] = (int(
                        bayer_image[row - 1, column - 1]) + int(
                        bayer_image[row + 1, column + 1])) / 2

            else:  # even row either G or B
                # |       |       |
                # |-------|-------|
                # |   G   |   B   |

                if column % 2 == 1:  # even column G
                    # |    |    |    |   read Blue Pixel Value
                    # |    |  B |    |
                    # |    |    |    |
                    new_image[row, column,
                              BLUE_CHANNEL] = bayer_image[row, column]
                    bayer_pattern[row, column, BLUE_CHANNEL] = 1

                    # | X  |     |  X |   interpolate red channel value
                    # |    |  B  |    |
                    # | X  |     |  X |
                    new_image[row, column, RED_CHANNEL] = (int(
                        bayer_image[row - 1, column - 1]) + int(
                        bayer_image[row + 1, column + 1]) + int(
                        bayer_image[row - 1, column + 1]) + int(
                        bayer_image[row + 1, column - 1])) / 4

                    # |    |  X  |    |   interpolate green channel value
                    # | X  |  B  |  X |
                    # |    |  X  |    |
                    new_image[row, column, GREEN_CHANNEL] = (int(
                        bayer_image[row - 1, column]) + int(
                        bayer_image[row + 1, column]) + int(
                        bayer_image[row, column + 1]) + int(
                        bayer_image[row, column - 1])) / 4
                else:  # odd column
                    # |    |    |    |   read Green Pixel Value
                    # |    |  G |    |
                    # |    |    |    |
                    new_image[row, column,
                              GREEN_CHANNEL] = bayer_image[row, column]
                    bayer_pattern[row, column, GREEN_CHANNEL] = 1

                    # |    |  X  |    |   interpolate red channel value
                    # |    |  G  |    |
                    # |    |  X  |    |
                    new_image[row, column, RED_CHANNEL] = (int(
                        bayer_image[row - 1, column]) + int(
                        bayer_image[row + 1, column])) / 2

                    # |    |     |    |   interpolate blue channel value
                    # | X  |  G  |  X |
                    # |    |     |    |
                    new_image[row, column, BLUE_CHANNEL] = (int(
                        bayer_image[row, column - 1]) + int(
                        bayer_image[row, column + 1])) / 2

    return new_image, bayer_pattern


def debayer_image_with_nearest_neighbor_per_indexing(
        bayer_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_image = np.zeros((bayer_image.shape[0], bayer_image.shape[1], 3))
    bayer_pattern = np.copy(new_image)

    # Again ignoring the border
    # reading the corresponding RGB values from their position in the bayer pattern
    # Position = Red Pixel (R1)
    new_image[1:-1, 1:-1, :][1::2, 1::2,
                             RED_CHANNEL] = bayer_image[1:-1, 1:-1][1::2, 1::2]
    new_image[1:-1, 1:-1, :][1::2, 1::2, GREEN_CHANNEL] = (
        bayer_image[:-2, 1:-1][1::2, 1::2].astype(int) +  # -1 0
        bayer_image[2:, 1:-1][1::2, 1::2].astype(int) +  # +1 0
        bayer_image[1:-1, :-2][1::2, 1::2].astype(int) +  # 0 -1
        bayer_image[1:-1, 2:][1::2, 1::2].astype(int)) / 4  # 0 +1
    new_image[1:-1, 1:-1, :][1::2, 1::2, BLUE_CHANNEL] = (
        bayer_image[:-2, :-2][1::2, 1::2].astype(int) +  # -1 -1
        bayer_image[:-2, 2:][1::2, 1::2].astype(int) +  # -1 +1
        bayer_image[2:, 2:][1::2, 1::2].astype(int) +  # +1 +1
        bayer_image[2:, :-2][1::2, 1::2].astype(int)) / 4  # +1 -1

    # Position = Green Pixel (G1)
    new_image[1:-1, 1:-1, :][
        ::2, 1::2, GREEN_CHANNEL] = bayer_image[1:-1, 1:-1][::2, 1::2]
    new_image[1:-1, 1:-1, :][::2, 1::2, RED_CHANNEL] = (
        bayer_image[2:, 1:-1][::2, 1::2].astype(int) +  # +1 0
        bayer_image[:-2, 1:-1][::2, 1::2].astype(int)) / 2  # -1 0
    new_image[1:-1, 1:-1, :][::2, 1::2, BLUE_CHANNEL] = (
        bayer_image[1:-1, 2:][::2, 1::2].astype(int) +  # 0 +1
        bayer_image[1:-1, :-2][::2, 1::2].astype(int)) / 2  # 0 -1

    # Position = Green Pixel (G2)
    new_image[1:-1, 1:-1, :][
        1::2, ::2, GREEN_CHANNEL] = bayer_image[1:-1, 1:-1][1::2, ::2]
    new_image[1:-1, 1:-1, :][1::2, ::2, RED_CHANNEL] = (
        bayer_image[1:-1, 2:][1::2, ::2].astype(int) +  # 0 +1
        bayer_image[1:-1, :-2][1::2, ::2].astype(int)) / 2  # 0 -1
    new_image[1:-1, 1:-1, :][1::2, ::2, BLUE_CHANNEL] = (
        bayer_image[2:, 1:-1][1::2, ::2].astype(int) +  # +1 0
        bayer_image[:-2, 1:-1][1::2, ::2].astype(int)) / 2  # -1 0

    # Position = Blue pixel (B1)
    new_image[1:-1, 1:-1, :][
        ::2, ::2, BLUE_CHANNEL] = bayer_image[1:-1, 1:-1][::2, ::2]
    new_image[1:-1, 1:-1, :][::2, ::2, GREEN_CHANNEL] = (
        bayer_image[:-2, 1:-1][::2, ::2].astype(int) +  # -1 0
        bayer_image[2:, 1:-1][::2, ::2].astype(int) +  # +1 0
        bayer_image[1:-1, :-2][::2, ::2].astype(int) +  # 0 -1
        bayer_image[1:-1, 2:][::2, ::2].astype(int)) / 4  # 0 +1
    new_image[1:-1, 1:-1, :][::2, ::2, RED_CHANNEL] = (
        bayer_image[:-2, :-2][::2, ::2].astype(int) +  # -1 -1
        bayer_image[:-2, 2:][::2, ::2].astype(int) +  # -1 + 1
        bayer_image[2:, 2:][::2, ::2].astype(int) +  # +1 +1
        bayer_image[2:, :-2][::2, ::2].astype(int)) / 4  # +1 -1

    # store bayer pattern for reference (due to the border ::2 -> 1::2 and vice versa
    # R = ::2, ::2, # G1 = 1::2, ::2, # G2 = ::2, 1::2, # B = 1::2, 1::2
    bayer_pattern[1:-1, 1:-1, :][1::2, 1::2, RED_CHANNEL] = 1
    bayer_pattern[1:-1, 1:-1, :][::2, 1::2, GREEN_CHANNEL] = 1
    bayer_pattern[1:-1, 1:-1, :][1::2, ::2, GREEN_CHANNEL] = 1
    bayer_pattern[1:-1, 1:-1, :][::2, ::2, BLUE_CHANNEL] = 1

    return new_image, bayer_pattern


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


def plot_debayered_image_and_pattern(
        debayered_image: np.ndarray,
        pattern: np.ndarray) -> None:
    image = debayered_image[1:-1, 1:-1, :]
    second_image = pattern[1:-1, 1:-1, :]

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='all')

    ax[0, 0].imshow(image[:, :, 0], cmap='gray')
    ax[0, 0].set_title('Red Channel')
    ax[0, 1].imshow(image[:, :, 1], cmap='gray')
    ax[0, 1].set_title('Green Channel')
    ax[0, 2].imshow(image[:, :, 2], cmap='gray')
    ax[0, 2].set_title('Blue Channel')

    print(np.max(image), np.min(image))
    ax[1, 1].imshow(np.array(image / 255.))
    ax[1, 1].set_title('Combined (R, G, B)')

    ax[1, 0].imshow(second_image)
    ax[1, 0].set_title('CFA: Color Filter Array')

    ax[1, 2].set_visible(False)
    plt.show()


@click.command()
@click.argument('filename', type=click.Path(exists=True))
def main(filename: Path) -> None:
    bayer = plt.imread(filename)

    start = time.time_ns()
    image, pattern = debayer_image_with_nearest_neighbor_iterative(
        bayer_image=bayer)
    end_iterative = time.time_ns()

    image, pattern = debayer_image_with_nearest_neighbor_per_indexing(
        bayer_image=bayer)
    end_indexing = time.time_ns()

    duration_iterative = end_iterative - start
    duration_index = end_indexing - end_iterative
    factor = duration_iterative / duration_index
    print(f'Speedup by Indexing instead of Looping {factor:.2f}x')
    # plt.imsave('./debayered.png', image/255.0)
    plot_debayered_image_and_pattern(image, pattern)


if __name__ == '__main__':
    main()
