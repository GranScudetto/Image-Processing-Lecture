import click
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from debayering import debayer_rgb_image_with_nearest_neighbor_per_indexing


def check_for_manipulation(suspicious_image: np.ndarray) -> None:
    image_debayered, _ = debayer_rgb_image_with_nearest_neighbor_per_indexing(
        suspicious_image)
    # check if the "bayer" correlation still holds true
    is_close = np.isclose(suspicious_image[:, :, :3], image_debayered[:, :, :], atol=5e-3)
    # set True to False and False to True to reflect the manipulation instead of that the values are close
    manipulated = np.invert(is_close)
    coordinates = np.sum(manipulated, axis=2)
    positions = np.zeros((suspicious_image.shape[0], suspicious_image.shape[1]))
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
