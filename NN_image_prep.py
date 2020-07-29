"""
This file implements an animal classifying neural network. Images of animals of different kinds are loaded and
pre-processed for a training set, validation set, and test set. An experimental ultra dense neural network is trained
from ultra_dense_neural_network.py. Hyper-parameters are then chosen with the validation set and final performance is
determined with the test set.
"""

import os
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from PIL import Image


def load_images(img_path, size=(28, 28), new_directory='new', re_size=False):
    """
    Loads animal images from raw-img folders
    :return: image_data -- list of numpy arrays containing image data
    """
    import load_jpeg_file as ld
    from random import shuffle

    categories = os.listdir(img_path)
    image_data = []
    label = 0

    print('Loading images')

    for category in categories:
        category_data = ld.load_image(img_path, category, label)
        if re_size:
            image_data_temp = resize_and_save(category_data, size, category, new_directory)
            image_data.extend(image_data_temp)
        else:
            image_data.extend(category_data)
        label += 1

#    shuffle(image_data)

    return image_data, label


def resize_and_save(image_data, size, category, directory):
    """
    Resizes images and saves into sub-folder with the name 'category'
    :param image_data: raw image data
    :param size: desired size for new images
    :param category: category of images being saved
    :param directory: starting point of new files
    :return: resized_images: list containing numpy arrays of resized images
    """
    # Set new parent folder for image categories
    start_path = os.getcwd() + '\\' + directory + '\\'

    resized_images = []

    # Create new directory for images of type 'category'
    try:
        new_path = start_path + '\\' + str(category)
        print("Creating new directory %s" % new_path)
        os.makedirs(new_path)
    except OSError:
        print("Creation of the directory %s failed" % new_path)
    else:
        print("Successfully created the directory %s" % new_path)

    print('Saving images of type: ' + category)
    # Resize and save images
    for i in tqdm(range(0, len(image_data))):
        # Resize image and append to list
        resized_image = resize(image_data[i][0], size)
        resized_images.append(resized_image)
        # Save new image to file
        im = Image.fromarray((resized_image * 255).astype(np.uint8))
        rgb_im = im.convert('RGB')
        filename = category + str(i) + '.jpg'
        filepath = new_path + '\\' + filename
        rgb_im.save(filepath, 'JPEG')

        #total_pixels = resized_image.shape[0]*resized_image.shape[1]*resized_image.shape[2]
        #assert(total_pixels == size[1] * 3)

    return resized_images


def split_data(image_data):
    """
    Implements data set splitting from scikit-learn to create a training, cross validation, and test set of data
    :param image_data: Shuffled list of images as numpy arrays and their corresponding labels
    :return: X_train, X_cross, X_test -- training, validation, and test data sets respectively
             Y_train, Y_cross, Y_test -- training, validation, and test labels respectively
    """
    from sklearn.model_selection import train_test_split

    X = []
    Y = []

    print('Splitting data into training, validation, and test sets')

    for i in tqdm(range(0, len(image_data))):
        X.append(image_data[i][0])
        Y.append(image_data[i][1])

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, shuffle=False, stratify=None)

    X_cross, X_test, Y_cross, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, shuffle=False, stratify=None)

    return X_train, np.asarray(Y_train), X_cross, np.asarray(Y_cross), X_test, np.asarray(Y_test)


def process_image(ini_X_train, ini_X_cross, ini_X_test):
    """
    Re-sizes images to be uniform using scikit_learn and reshapes images into vectors for input into network

    :param ini_X_train: Training, validation, and test sets of data and labels
    :param ini_X_cross: see above
    :param ini_X_test: see above
    :return: Returns reshaped versions of input
    """

    X_train_temp = []
    X_cross_temp = []
    X_test_temp = []

    print('Reshaping X_train set')
    for i in tqdm(range(0, len(ini_X_train))):
        X_train_temp.append(ini_X_train[i].reshape(1, -1))

    X_train = np.concatenate(X_train_temp, axis=0)

    print('Reshaping X_cross set')
    for i in tqdm(range(0, len(ini_X_cross))):
        X_cross_temp.append(ini_X_cross[i].reshape(1, -1))

    X_cross = np.concatenate(X_cross_temp, axis=0)

    print('Reshaping X_test set')
    for i in tqdm(range(0, len(ini_X_test))):
        X_test_temp.append(ini_X_test[i].reshape(1, -1))

    X_test = np.concatenate(X_test_temp, axis=0)

    return X_train, X_cross, X_test


def process_mnist(ini_X_train):
    """
    Re-sizes images to be uniform using scikit_learn and reshapes images into vectors for input into network

    :param ini_X_train: Training, validation, and test sets of data and labels
    :return: Returns reshaped versions of input
    """

    X_train_temp = []
    print('Reshaping X_train set')
    for i in tqdm(range(0, len(ini_X_train))):
        X_train_temp.append(ini_X_train[i][0].reshape(1, -1))

    X_set = np.concatenate(X_train_temp, axis=0)

    return X_set


def process_softmax_Y(Y, max_label):
    """
    Converts Y label into a vector appropriate for the softmax classification
    :param Y: Y label vector
    :param max_label: The largest integer value representing a label in a dat set
    :return: Y_soft: The softmax label vector
    """
    num_labels = max_label
    m = len(Y)
    Y_soft = np.zeros((m, num_labels))

    for i in range(0, m):
        Y_soft[i, Y[i][1]] = 1

    return Y_soft


def get_images(size=(28, 28), new_directory='new', re_size=False):
    """
    Gets image data from a folder set in load_jpeg_file.py and sorts them into Training, Cross-Validation, and Test sets
    for training a neural network. Returns these input data sets where each image has been reshaped into a vector.
    :param size: size of resize of images if resizing is needed
    :param new_directory: sets new directory name to save resized images in
    :param re_size: True or False tells program whether to resize images
    :return: X_train, Y_train, X_cross, Y_cross, X_test, Y_test: Data sets for training, validating, and testing neural
            network with learned parameters
    """
    img_path = 'C:/Users/Dawson/PycharmProjects/Deep Learning Projects/resized-img/'

    # Load images and resize if needed. Data is returned in random order.
    image_data = load_images(img_path, size, new_directory, re_size)

    # Split data into training, validation, and test sets
    X_train_whole, Y_train, X_cross_whole, Y_cross, X_test_whole, Y_test = split_data(image_data)

    # Reshape data into vectors for processing by neural network
    X_train_flat, X_cross_flat, X_test_flat = process_image(X_train_whole, X_cross_whole, X_test_whole)

    # Normalize image data from 0 - 1 instead of RGB 0-255
    X_train = X_train_flat/255.
    X_cross = X_cross_flat/255.
    X_test = X_test_flat/255.

    return X_train, Y_train, X_cross, Y_cross, X_test, Y_test


def get_mnist_data(size=(28, 28)):
    """
    Loads MNIST handwritten digits train and test sets into arrays and returns arrays
    :param size: size of the MNIST set
    :return: X_train, Y_train, X_test, Y_test
    """
    train_path = 'C:/Users/Dawson/PycharmProjects/Deep Learning Projects/MNIST Set/train/'
    test_path = 'C:/Users/Dawson/PycharmProjects/Deep Learning Projects/MNIST Set/test/'

    train_image_data, max_label = load_images(train_path, size)
    X_train = process_mnist(train_image_data)
    Y_train = process_softmax_Y(train_image_data, max_label)

    test_image_data = load_images(test_path, size)
    X_test = process_mnist(test_image_data[0])
    Y_test = process_softmax_Y(test_image_data[0], max_label)

    X_train = X_train/255
    X_test = X_test/255

    return X_train, Y_train, X_test, Y_test
