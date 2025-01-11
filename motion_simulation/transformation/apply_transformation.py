import numpy as np
from scipy.ndimage.interpolation import affine_transform

def apply_affine_transform(image, transform_matrix, order, channel_wise = False, channel_index = 0, fill_mode="constant", cval=0.):
    
    if channel_wise == False:
        final_affine_matrix = transform_matrix[: image.ndim, : image.ndim]
        final_offset = transform_matrix[: image.ndim, image.ndim]

        return affine_transform(image, final_affine_matrix, final_offset, order=order, mode=fill_mode, cval=cval)

    else:
        image = np.rollaxis(image, channel_index, 0)

        array = [
            apply_affine_transform(
                x_channel, transform_matrix, fill_mode=fill_mode, cval=cval, order=order
            )
            for x_channel in image
        ]
        array = np.stack(array, axis=0)
        return np.rollaxis(array, 0, channel_index + 1)
#

