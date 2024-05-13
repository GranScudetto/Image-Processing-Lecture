import matplotlib.pyplot as plt
import numpy as np
from one_dimensional_filter import rgb2gray


def sobel_filter(image: np.ndarray, threshold: int) -> np.ndarray:
    result = np.zeros_like(image)
    sobel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])

    for row in range(1, image.shape[0] - 1):
        for column in range(1, image.shape[1] - 1):
            horizontal = (sobel_x[0, 0] * image[row - 1, column - 1] +
                          sobel_x[0, 1] * image[row - 1, column] +
                          sobel_x[0, 2] * image[row - 1, column + 1] +

                          sobel_x[1, 0] * image[row, column - 1] +
                          sobel_x[1, 1] * image[row, column] +
                          sobel_x[1, 2] * image[row, column + 1] +

                          sobel_x[2, 0] * image[row + 1, column - 1] +
                          sobel_x[2, 1] * image[row + 1, column] +
                          sobel_x[2, 2] * image[row + 1, column + 1])

            vertical = (sobel_y[0, 0] * image[row - 1, column - 1] +
                        sobel_y[0, 1] * image[row - 1, column] +
                        sobel_y[0, 2] * image[row - 1, column + 1] +

                        sobel_y[1, 0] * image[row, column - 1] +
                        sobel_y[1, 1] * image[row, column] +
                        sobel_y[1, 2] * image[row, column + 1] +

                        sobel_y[2, 0] * image[row + 1, column - 1] +
                        sobel_y[2, 1] * image[row + 1, column] +
                        sobel_y[2, 2] * image[row + 1, column + 1])

            result[row, column] = np.sqrt(
                np.square(horizontal) + np.square(vertical))

    return np.where(result > threshold, 1, 0)


def laplace_filter(image: np.ndarray, threshold: int) -> np.ndarray:
    result = np.zeros_like(image)
    laplace = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

    for row in range(1, image.shape[0] - 1):
        for column in range(1, image.shape[1] - 1):
            result[row, column] = abs(laplace[0, 0] * image[row - 1, column - 1] +
                                      laplace[0, 1] * image[row - 1, column] +
                                      laplace[0, 2] * image[row - 1, column + 1] +

                                      laplace[1, 0] * image[row, column - 1] +
                                      laplace[1, 1] * image[row, column] +
                                      laplace[1, 2] * image[row, column + 1] +

                                      laplace[2, 0] * image[row + 1, column - 1] +
                                      laplace[2, 1] * image[row + 1, column] +
                                      laplace[2, 2] * image[row + 1, column + 1])

    return np.where(result > threshold, 1, 0)


if __name__ == '__main__':
    test_image = plt.imread('../data/OpenCVExamples/messi5.jpg')

    if test_image.ndim == 3:
        test_image = rgb2gray(test_image)

    threshold_value = 100
    laplace_result = laplace_filter(test_image, threshold_value)
    sobel_result = sobel_filter(test_image, threshold_value)

    fig, ax = plt.subplots(1, 3, sharey='all', sharex='all')
    ax[0].imshow(test_image, cmap='gray')
    ax[1].imshow(sobel_result, cmap='gray')
    ax[2].imshow(laplace_result, cmap='gray')
    plt.show()
