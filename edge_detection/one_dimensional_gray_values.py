import matplotlib.pyplot as plt
import numpy as np

HEIGHT = 300


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def histogram_of_gray_values(image_array: np.ndarray) -> None:
    gray_vals = image_array.reshape((-1))
    plt.hist(gray_vals, bins=np.arange(2**8))
    plt.show()


def gray_value_along_h_line(image: np.ndarray, row_index: int) -> None:
    highlighted_image = np.copy(image)

    fig, ax = plt.subplots(3, 1, sharex='all')
    row_values = image[row_index, :]
    # for visualization highlighting the selected row (5 rows wide)
    highlighted_image[row_index-2:row_index+2, :] = 255
    ax[0].imshow(highlighted_image[row_index -
                 HEIGHT:row_index+HEIGHT, :], cmap='gray')
    ax[1].plot(row_values)
    # forward_diff = forward_difference(row_values)
    # ax[2].plot(forward_diff)
    # backward_diff = backward_difference(row_values)
    # ax[3].plot(backward_diff)
    central_diff = central_difference(row_values)
    ax[2].plot(central_diff)
    plt.show()


def forward_difference(values: np.ndarray) -> np.ndarray:
    # formula (f(x+1) - f(x))/h (h = 1 here)
    x = values[:-1]
    x_plus_1 = values[1:]
    return np.subtract(x_plus_1, x)


def backward_difference(values: np.ndarray) -> np.ndarray:
    # formula (f(x) - f(x-1))/h
    x = values[1:]
    x_minus_1 = values[:-1]
    return np.subtract(x, x_minus_1)


def central_difference(values: np.ndarray):
    # formula (f(x+1) - f(x-1))/2h
    return 0.5 * np.add(forward_difference(values), backward_difference(values))


if __name__ == '__main__':
    image = plt.imread('../data/20230615-7R500254g_scaled.jpg')
    image = rgb2gray(image)
    # histogram_of_gray_values(image)
    gray_value_along_h_line(image, int(4702/2))
