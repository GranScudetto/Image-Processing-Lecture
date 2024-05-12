import numpy as np
import matplotlib.pyplot as plt


COLOR_FILTER_TO_CHANNEL = {'R': 0, 'G': 1, 'B': 2}


def get_color_filter_array(pattern: str, array: np.ndarray) -> np.ndarray:
    bayer = np.zeros((array.shape[0], array.shape[1], 3))
    # | 00 | 10 | -> | R | G |
    # | 01 | 11 |    | G | B |

    #  odd rows, odd columns | X |   |
    #                        |   |   |
    bayer[::2, ::2, COLOR_FILTER_TO_CHANNEL[pattern[0]]] = 255  # f.e. Red (RGGB)

    # odd rows, even columns |   | X |
    #                        |   |   |
    bayer[::2, 1::2, COLOR_FILTER_TO_CHANNEL[pattern[1]]] = 255  # f.e. Green (RGGB)

    # even rows, odd columns |   |   |
    #                        | X |   |
    bayer[1::2, ::2, COLOR_FILTER_TO_CHANNEL[pattern[2]]] = 255  # f.e. Green (RGGB)

    # even rows, even columns |   |   |
    #                         |   | X |
    bayer[1::2, 1::2, COLOR_FILTER_TO_CHANNEL[pattern[3]]] = 255  # f.e. Blue (RGGB)

    return bayer


if __name__ == "__main__":
    rggb_pattern = 'RGGB'
    rgbr_pattern = 'RGBR'
    grgb_pattern = 'GRGB'

    color_filter_array = np.zeros((16, 16))  # has to be a multiple of 2!
    rggb = get_color_filter_array(rggb_pattern, color_filter_array)
    rgbr = get_color_filter_array(rgbr_pattern, color_filter_array)
    grgb = get_color_filter_array(grgb_pattern, color_filter_array)

    fig, (filter_array_1, filter_array_2, filter_array_3) = plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all')
    filter_array_1.imshow(rggb)
    filter_array_1.set_title(rggb_pattern)

    filter_array_2.imshow(rgbr)
    filter_array_2.set_title(rgbr_pattern)

    filter_array_3.imshow(grgb)
    filter_array_3.set_title(grgb_pattern)

    plt.show()
