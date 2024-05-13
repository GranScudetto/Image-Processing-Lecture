import click
import matplotlib.pyplot as plt
import numpy as np

COLOR_FILTER_TO_CHANNEL = {'R': 0, 'G': 1, 'B': 2}


def get_color_filter_array(pattern: str, array: np.ndarray) -> np.ndarray:
    bayer = np.zeros((array.shape[0], array.shape[1], 3))
    # | 00 | 10 | -> | R | G |
    # | 01 | 11 |    | G | B |

    #  odd rows, odd columns | X |   |
    #                        |   |   |
    # f.e. Red (RGGB)
    bayer[::2, ::2, COLOR_FILTER_TO_CHANNEL[pattern[0]]] = 255

    # odd rows, even columns |   | X |
    #                        |   |   |
    # f.e. Green (RGGB)
    bayer[::2, 1::2, COLOR_FILTER_TO_CHANNEL[pattern[1]]] = 255

    # even rows, odd columns |   |   |
    #                        | X |   |
    # f.e. Green (RGGB)
    bayer[1::2, ::2, COLOR_FILTER_TO_CHANNEL[pattern[2]]] = 255

    # even rows, even columns |   |   |
    #                         |   | X |
    # f.e. Blue (RGGB)
    bayer[1::2, 1::2, COLOR_FILTER_TO_CHANNEL[pattern[3]]] = 255

    return bayer


@click.command()
@click.argument('pattern', type=click.STRING, default='RGGB')
@click.argument('dimension', type=click.INT)
def interactive_color_filter(pattern: str, dimension: int) -> None:
    assert len(pattern) == 4
    assert dimension % 2 == 0
    color_filter = np.zeros((dimension, dimension))
    mosaic = get_color_filter_array(pattern, color_filter)
    plt.imshow(mosaic)
    plt.show()


def plot_group_of_color_filters() -> None:
    rggb_pattern = 'RGGB'
    rgbr_pattern = 'RGBR'
    grgb_pattern = 'GRGB'

    color_filter_array = np.zeros((16, 16))  # has to be a multiple of 2!
    rggb = get_color_filter_array(rggb_pattern, color_filter_array)
    rgbr = get_color_filter_array(rgbr_pattern, color_filter_array)
    grgb = get_color_filter_array(grgb_pattern, color_filter_array)

    fig, (filter_array_1, filter_array_2, filter_array_3) = plt.subplots(
        nrows=1, ncols=3, sharex='all', sharey='all')
    filter_array_1.imshow(rggb.astype(int))
    filter_array_1.set_title(rggb_pattern)

    filter_array_2.imshow(rgbr.astype(int))
    filter_array_2.set_title(rgbr_pattern)

    filter_array_3.imshow(grgb.astype(int))
    filter_array_3.set_title(grgb_pattern)

    plt.show()


if __name__ == "__main__":
    # plot_group_of_color_filters()
    interactive_color_filter()
