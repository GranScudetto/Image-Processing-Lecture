from typing import Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2


def debayer_image_iterative(bayer_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    new_image = np.zeros((bayer.shape[0], bayer.shape[1], 3))
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

            if row % 2 == 0:  # odd row (starts with index 0 (0, 2, 4, 6, ...)) => either R or G
                # |   R   |   G   |
                # |-------|-------|
                # |       |       |

                if column % 2 == 1:  # even column (1, 3, ...)
                    # |    |    |    |   read Green Pixel Value
                    # |    |  G |    |
                    # |    |    |    |
                    new_image[row, column, GREEN_CHANNEL] = bayer_image[row, column]
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
                    new_image[row, column, RED_CHANNEL] = bayer_image[row, column]
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
                    new_image[row, column, BLUE_CHANNEL] = bayer_image[row, column]
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
                    new_image[row, column, GREEN_CHANNEL] = bayer_image[row, column]
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
                        bayer_image[row, column-1]) + int(
                        bayer_image[row, column+1])) / 2

    return new_image, bayer_pattern


def debayer_image_per_indexing(bayer_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_image = np.zeros((bayer.shape[0], bayer.shape[1], 3))
    bayer_pattern = np.copy(new_image)

    # Again ignoring the border
    # reading the corresponding RGB values from their position in the bayer pattern
    # Position = Red Pixel (R1)
    new_image[1:-1, 1:-1, :][1::2, 1::2, RED_CHANNEL] = bayer_image[1:-1, 1:-1][1::2, 1::2]
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
    new_image[1:-1, 1:-1, :][::2, 1::2, GREEN_CHANNEL] = bayer_image[1:-1, 1:-1][::2, 1::2]
    new_image[1:-1, 1:-1, :][::2, 1::2, RED_CHANNEL] = (
                        bayer_image[2:, 1:-1][::2, 1::2].astype(int) +  # +1 0
                        bayer_image[:-2, 1:-1][::2, 1::2].astype(int)) / 2  # -1 0
    new_image[1:-1, 1:-1, :][::2, 1::2, BLUE_CHANNEL] = (
            bayer_image[1:-1, 2:][::2, 1::2].astype(int) +  # 0 +1
            bayer_image[1:-1, :-2][::2, 1::2].astype(int)) / 2  # 0 -1

    # Position = Green Pixel (G2)
    new_image[1:-1, 1:-1, :][1::2, ::2, GREEN_CHANNEL] = bayer_image[1:-1, 1:-1][1::2, ::2]
    new_image[1:-1, 1:-1, :][1::2, ::2, RED_CHANNEL] = (
                        bayer_image[1:-1, 2:][1::2, ::2].astype(int) +  # 0 +1
                        bayer_image[1:-1, :-2][1::2, ::2].astype(int)) / 2  # 0 -1
    new_image[1:-1, 1:-1, :][1::2, ::2, BLUE_CHANNEL] = (
        bayer[2:, 1:-1][1::2, ::2].astype(int) +  # +1 0
        bayer[:-2, 1:-1][1::2,::2].astype(int)  # -1 0
    ) / 2

    # Position = Blue pixel (B1)
    new_image[1:-1, 1:-1, :][::2, ::2, BLUE_CHANNEL] = bayer_image[1:-1, 1:-1][::2, ::2]
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


def plot_debayered_image_and_pattern(debayered_image: np.ndarray, pattern: np.ndarray) -> None:
    image = debayered_image[1:-1, 1:-1, :]
    second_image = pattern[1:-1, 1:-1, :]

    fig, ax = plt.subplots(2, 3, sharex='all', sharey='all')

    ax[0, 0].imshow(image[:, :, 0], cmap='gray')
    ax[0, 0].set_title('Red Channel')
    ax[0, 1].imshow(image[:, :, 1], cmap='gray')
    ax[0, 1].set_title('Green Channel')
    ax[0, 2].imshow(image[:, :, 2], cmap='gray')
    ax[0, 2].set_title('Blue Channel')

    print(np.max(image), np.min(image))
    ax[1, 1].imshow(np.array(image/255.))
    ax[1, 1].set_title('Combined (R, G, B)')

    ax[1, 0].imshow(second_image)
    ax[1, 0].set_title('CFA: Color Filter Array')

    ax[1, 2].set_visible(False)
    plt.show()


if __name__ == '__main__':
    # bayer_file_path = '/home/uia59450/Schreibtisch/Lehrauftrag DHBW/2.bmp'
    bayer_file_path = '/Volumes/Macintosh HD/Users/matthiasnacken/Desktop/Lehrauftrag DHBW/T3ES9008_Bilddatenverarbeitung_und_Mustererkennung/Materialien/1.bmp'

    bayer = plt.imread(bayer_file_path)
    # image, pattern = debayer_image_iterative(bayer_image=bayer)
    image, pattern = debayer_image_per_indexing(bayer_image=bayer)
    plot_debayered_image_and_pattern(image, pattern)
