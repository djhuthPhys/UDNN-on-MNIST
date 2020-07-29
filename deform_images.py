import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def deform_images(images, size, alpha, sigma, seed):
    """
    Deforms images using elastic image deformation and randomized affine transforms
    :param images: list of images in the shape (number of images, number of pixels per image)
    :param size: size of images for reshaping into 2D matrix
    :param alpha: parameter for elastic deformation
    :param sigma: parameter for elastic deformation
    :param gamma: parameter defining scaling range
    :param theta: parameter defining rotation range
    :return: images: list of deformed images in the shape (number of images, number of pixels per image)
    """
    np.random.seed(seed)

    # Reshape images for processing
    images = np.reshape(images, (-1, size, size))

    # Elastic deformation of each image
    alpha_rand = np.random.uniform(alpha-20, alpha+20)
    sigma_rand = np.random.uniform(sigma-1, sigma+1)
    new_images = elastic_transform(images, alpha_rand, sigma_rand, seed)

    # Rotate image by theta
    #for i in range(0, images.shape[0]):
    #random_theta = 2 * (theta * np.random.random((images.shape[0], 1, 1)) - theta/2)
    #rotated_image = transform.rotate(images, random_theta)
    #deformed_images[i, :, :] = rotated_image

    # Reshape images
    images = np.reshape(new_images, (-1, size**2))

    return images


def elastic_transform(image, alpha, sigma, seed):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    np.random.seed(seed)

    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    z, x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    indices = (z, x+dx, y+dy)

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')

    return distorted_image