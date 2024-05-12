import numpy as np
import matplotlib.pyplot as plt

from debayering import debayer_rgb_image_with_nearest_neighbor_per_indexing

image = plt.imread('debayered3.png')
# plt.imshow(image[:, :, 0], cmap='gray')
# plt.show()
print(image.shape)
print(image[:10, :10, :])

image_debayered, pattern = debayer_rgb_image_with_nearest_neighbor_per_indexing(image)
print(image_debayered.shape)
# plt.imshow(image_debayered)
# plt.show()


is_close = np.isclose(image[:, :, :3], image_debayered[:, :, :], atol=1e-2)
manipulated = np.invert(is_close)
plt.imshow(manipulated.astype(int)*255, cmap='gray')
plt.show()
print((np.sum(is_close)/3)/image[3:-3,:3].shape[0])

