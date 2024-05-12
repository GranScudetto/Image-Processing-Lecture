import numpy as np
import matplotlib.pyplot as plt


def stripe_pattern(size: int) -> np.ndarray:
    array = np.zeros((size, size))
    array[::2, :] = 1
    return array


def gray_scale(bit_depth:int=8) -> np.ndarray:
    max_val = 2**bit_depth
    array = np.zeros((max_val, max_val))
    for column in range(array.shape[1]):
        array[:, column] = column
    return array


def checkerboard() -> np.ndarray:
    board = np.zeros((8, 8))
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if (i + j) % 2 == 0:
                board[i, j] = 1
    return board


def black_image_with_white_rectangle() -> np.ndarray:
    image = np.zeros((5, 5))
    image[2, 2] = 1
    return image


if __name__ == '__main__':

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].imshow(stripe_pattern(8), cmap='gray')
    ax[0, 0].set_title('Stripes')

    ax[0, 1].imshow(gray_scale(12), cmap='gray')
    ax[0, 1].set_title('GrayScale')

    ax[1, 0].imshow(checkerboard(), cmap='gray')
    ax[1, 0].set_title('Checkerboard')

    ax[1, 1].imshow(black_image_with_white_rectangle(), cmap='gray')
    ax[1, 1].set_title('White box, black background')
    plt.tight_layout()  # Magic trick to read the subtitles comfortably
    plt.show()
