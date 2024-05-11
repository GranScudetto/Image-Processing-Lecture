import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2

#bayer_file_path = '/home/uia59450/Schreibtisch/Lehrauftrag DHBW/2.bmp'
bayer_file_path = '/Volumes/Macintosh HD/Users/matthiasnacken/Desktop/Lehrauftrag DHBW/T3ES9008_Bilddatenverarbeitung_und_Mustererkennung/Materialien/1.bmp'

bayer = plt.imread(bayer_file_path)
# img = cv2.imread(bayer_file_path, cv2.IMREAD_GRAYSCALE)

RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2


new_image = np.zeros((bayer.shape[0], bayer.shape[1], 3))
second_image = np.copy(new_image)


for row in range(1, bayer.shape[0] - 1):  # iterate rows
    for column in range(1, bayer.shape[1] - 1):  # iterate columns
        #       columns ->
        #  rows |   R   |   G   |
        #   |   |-------|-------|
        #  \|/  |   G   |   B   |

        if row % 2 == 0:  # odd row (starts with index 0) => either R or G
            # |   R   |   G   |
            # |-------|-------|
            # |       |       |

            if column % 2 == 1:  # even column (1, 3, ...)
                # |    |    |    |   read Green Pixel Value
                # |    |  G |    |
                # |    |    |    |
                new_image[row, column, GREEN_CHANNEL] = bayer[row, column]
                second_image[row, column, GREEN_CHANNEL] = bayer[row, column]

                # |    |     |    |   interpolate red channel value
                # | X  |  G  |  X |
                # |    |     |    |
                new_image[row, column, RED_CHANNEL] = (int(
                    bayer[row, column - 1]) + int(
                    bayer[row, column + 1])) / 2

                # |    |  X  |    |   interpolate blue channel value
                # |    |  G  |    |
                # |    |  X  |    |
                new_image[row, column, BLUE_CHANNEL] = (int(
                    bayer[row - 1, column]) + int(
                    bayer[row + 1, column])) / 2

            else:
                # |    |    |    |   read Red Pixel Value
                # |    |  R |    |
                # |    |    |    |
                new_image[row, column, RED_CHANNEL] = bayer[row, column]
                second_image[row, column, RED_CHANNEL] = bayer[row, column]

                # |    |  X  |    |   interpolate green channel value
                # | X  |  R  |  X |
                # |    |  X  |    |
                new_image[row, column, GREEN_CHANNEL] = (int(
                    bayer[row - 1, column]) + int(
                    bayer[row + 1, column]) + int(
                    bayer[row, column + 1]) + int(
                    bayer[row, column - 1])) / 4

                # | X  |     |    |   interpolate blue channel value
                # |    |  R  |    |
                # |    |     |  X |
                new_image[row, column, BLUE_CHANNEL] = (int(
                    bayer[row - 1, column - 1]) + int(
                    bayer[row + 1, column + 1])) / 2

        else:  # even row either G or B
            # |       |       |
            # |-------|-------|
            # |   G   |   B   |

            if column % 2 == 1:  # even column G
                # |    |    |    |   read Blue Pixel Value
                # |    |  B |    |
                # |    |    |    |
                new_image[row, column, BLUE_CHANNEL] = bayer[row, column]
                second_image[row, column, BLUE_CHANNEL] = bayer[row, column]

                # | X  |     |  X |   interpolate red channel value
                # |    |  B  |    |
                # | X  |     |  X |
                new_image[row, column, RED_CHANNEL] = (int(
                    bayer[row - 1, column - 1]) + int(
                    bayer[row + 1, column + 1]) + int(
                    bayer[row - 1, column + 1]) + int(
                    bayer[row + 1, column - 1])) / 4

                # |    |  X  |    |   interpolate green channel value
                # | X  |  B  |  X |
                # |    |  X  |    |
                new_image[row, column, GREEN_CHANNEL] = (int(
                    bayer[row - 1, column]) + int(
                    bayer[row + 1, column]) + int(
                    bayer[row, column + 1]) + int(
                    bayer[row, column - 1])) / 4
            else:  # odd column
                # |    |    |    |   read Green Pixel Value
                # |    |  G |    |
                # |    |    |    |
                new_image[row, column, GREEN_CHANNEL] = bayer[row, column]
                second_image[row, column, GREEN_CHANNEL] = bayer[row, column]

                # |    |  X  |    |   interpolate red channel value
                # |    |  G  |    |
                # |    |  X  |    |
                new_image[row, column, RED_CHANNEL] = (int(
                    bayer[row - 1, column]) + int(
                    bayer[row + 1, column])) / 2

                # |    |     |    |   interpolate blue channel value
                # | X  |  G  |  X |
                # |    |     |    |
                new_image[row, column, BLUE_CHANNEL] = (int(
                    bayer[row, column-1]) + int(
                    bayer[row, column+1])) / 2


image = new_image[1:-1, 1:-1, :]
second_image = second_image[1:-1, 1:-1, :]

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
