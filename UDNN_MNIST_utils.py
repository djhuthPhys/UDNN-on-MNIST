import math
import numpy as np
import pickle
import deform_images as di
import NN_image_prep as imprep
from tqdm import tqdm

import wandb

hyperparameter_defaults = dict(
    layer_sizes=[784, 256, 256, 256, 256, 10],
    learning_rate=0.003,
    num_epochs=300,
    mini_batch_size=64,
    alpha=70,
    sigma=4,
    keep_prob=1,
    network_type='UDNN'
    )

run = wandb.init(config=hyperparameter_defaults, project="MNIST - FNN-UDNN Comparison")
config = wandb.config


def initialize_dense_parameters(layer_sizes):
    """ Randomly initializes the weights in W and zeros the biases in b using the size of the layers
        Arguments:
            layer_sizes -- vector where each element l represents the number of hidden units in layer l

        Returns:
            parameters -- dictionary of randomly initialized weights and zeroed biases
    """
    parameters = {}
    L = len(layer_sizes)
    np.random.seed(0)

    for l in range(0, L):
        for k in range(l+1, L):
            parameters['W' + str(l) + ',' + str(k)] = np.random.uniform(-0.5, 0.5, (layer_sizes[k], layer_sizes[l])) \
                                                          * np.sqrt(2/layer_sizes[l])
            parameters['b' + str(l) + ',' + str(k)] = np.zeros((layer_sizes[k], 1))

            assert (parameters['W' + str(l) + ',' + str(k)].shape == (layer_sizes[k], layer_sizes[l]))
            assert (parameters['b' + str(l) + ',' + str(k)].shape == (layer_sizes[k], 1))

    return parameters


def initialize_dense_adam(layer_sizes):
    """
    Initializes a dictionary with keys: 'dW0,1...dWL-1,L' and 'db0,1...dbL-1,L' and values of zero matrices the same
    size as the corresponding parameters
    :param layer_sizes: dictionary with parameter values
    :return: v: dictionary of initialized velocity values
    """

    v = {}
    s = {}
    L = len(layer_sizes)

    for l in range(0, L):
        for k in range(l+1, L):
            v['dW' + str(l) + ',' + str(k)] = np.zeros((layer_sizes[k], layer_sizes[l]))
            v['db' + str(l) + ',' + str(k)] = np.zeros((layer_sizes[k], 1))
            s['dW' + str(l) + ',' + str(k)] = np.zeros((layer_sizes[k], layer_sizes[l]))
            s['db' + str(l) + ',' + str(k)] = np.zeros((layer_sizes[k], 1))

            assert (v['dW' + str(l) + ',' + str(k)].shape == (layer_sizes[k], layer_sizes[l]))
            assert (v['db' + str(l) + ',' + str(k)].shape == (layer_sizes[k], 1))
            assert (s['dW' + str(l) + ',' + str(k)].shape == (layer_sizes[k], layer_sizes[l]))
            assert (s['db' + str(l) + ',' + str(k)].shape == (layer_sizes[k], 1))

    return v, s


def create_mini_batches(X, Y, mini_batch_size, seed):
    """
    Creates and returns a randomized list of num_batches mini-batches out of the data set (X,Y) using seed to randomize.
    :param X: numpy array of input data of size (number of features, number of examples)
    :param Y: numpy array of  labels for X of size (number of examples, labels)
    :param mini_batch_size: defines the number of examples from (X,Y) for each mini-batch
    :return:mini_batches
    """

    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    # Shuffle X and Y
    permutation = list(np.random.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[permutation, :]

    num_complete_mini_batches = math.floor(m/mini_batch_size)
    for i in range(0, num_complete_mini_batches):
        mini_batch_X = X_shuffled[:, i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y = Y_shuffled[i*mini_batch_size:(i+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = X_shuffled[:, mini_batch_size*num_complete_mini_batches-1:-1]
        mini_batch_Y = Y_shuffled[mini_batch_size*num_complete_mini_batches-1:-1]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def leaky_relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.where(Z > 0, Z, Z * 0.01)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def leaky_relu_backward(dA, cache):
    """ Implement the backward propagation for a single RELU unit.
        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.ones_like(Z)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0.01
    dZ = np.multiply(dA, dZ)

    assert (dZ.shape == Z.shape)

    return dZ


def softmax(Z):
    """
    Calculates softmax from the the linear term Z
    :param Z: numpy array of linear terms in network layer
    :return: A, cache: activation layer A and cache containing Z
    """

    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / np.sum(e_Z, axis=0, keepdims=True)

    cache = Z

    return A, cache


def softmax_backward(AL, Y):
    """
    Calculates dZ with the gradient of the softmax function
    :param AL: matrix of the last activation values in the network
    :param Y: the labels of each example
    :return: dZ: derivative of the softmax function with respect to Z
    """

    dZ = AL - Y.T

    return dZ


def forward_linear(A, W, b):
    """ Performs forward propagation calculation with relevant layers activation value, weights, and biases
        Arguments:
            A -- activation vector from relevant layer (size of previous layer, num of examples)
            W -- weights vector from relevant layer (size of current layer, size of previous layer)
            b -- bias vector from relevant activation (size of current layer, 1)

        Returns:
            Z -- input for the activation function in the next layer (size of current layer, num of examples)
            cache -- tuple containing 'A', 'W', and 'b' for efficient backward propagation
    """

    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache


def dense_forward_prop(X, parameters, layer_sizes):

    """ Implements forward propagation for an ultra dense neural network
        Arguments:
            X -- Input data vector
            parameters -- vector of randomly initialized weights
            layer_sizes -- vector of number of hidden units in each layer

        Returns:
            A -- last activation value in network
            caches -- list of every cache from forward_activation (linear_cache, activation_cache)
    """

    caches = {}
    A_rel, caches['A0'] = X, X
    L = len(layer_sizes)

    # Implementation of dense forward propagation - Outer loop loops over layers of network to calculate final
    # activation values for the hidden units in layer l. Inner loop loops over all preceding layers, k, and accumulates
    # the total linear value, Z. A and Z values are cached in dictionary for access during back propagation.

    for l in range(1, L-1):
        Z = 0
        for k in range(0, l):
            A_rel = caches['A' + str(k)]
            Z_temp, linear_cache = forward_linear(A_rel,  parameters['W' + str(k) + ',' + str(l)],
                                                  parameters['b' + str(k) + ',' + str(l)])
            Z += Z_temp
            caches['Z' + str(l)] = Z

        A, activation_cache = leaky_relu(Z)
        caches['A' + str(l)] = A

    Z = 0
    for k in range(0, L-1):
        A_rel = caches['A' + str(k)]
        Z_temp, linear_cache = forward_linear(A_rel, parameters['W' + str(k) + ',' + str(L-1)],
                                              parameters['b' + str(k) + ',' + str(L-1)])
        Z += Z_temp
        caches['Z' + str(L-1)] = Z

    A, activation_cache = softmax(Z)
    caches['A' + str(L-1)] = A

    # assert (A.shape == (1, X.shape[1]))

    return A, caches


def dense_dropout_forward_prop(X, parameters, keep_prob, layer_sizes):

    """ Implements forward propagation for an ultra dense neural network with dropout regularization
        Arguments:
            X -- Input data vector
            parameters -- vector of randomly initialized weights
            layer_sizes -- vector of number of hidden units in each layer
            keep_prob -- defines the probability a node will be kept in the activation process

        Returns:
            A -- last activation value in network
            caches -- list of every cache from forward_activation (linear_cache, activation_cache)
            dropouts -- list of dropout arrays to be used in backward prop
    """

    caches = {}
    dropouts = {}
    A_rel, caches['A0'] = X, X
    L = len(layer_sizes)

    # Implementation of dense forward propagation - Outer loop loops over layers of network to calculate final
    # activation values for the hidden units in layer l. Inner loop loops over all preceding layers, k, and accumulates
    # the total activation value. A and Z values are cached in dictionary for access during back propagation.

    for l in range(1, L-1):
        Z = 0
        for k in range(0, l):
            A_rel = caches['A' + str(k)]
            Z_temp, linear_cache = forward_linear(A_rel,  parameters['W' + str(k) + ',' + str(l)],
                                                  parameters['b' + str(k) + ',' + str(l)])
            Z += Z_temp

        A_temp, activation_cache = leaky_relu(Z)
        dropout = (np.random.rand(A_temp.shape[0], A_temp.shape[1]) < keep_prob)
        A = (np.multiply(A_temp, dropout)) / keep_prob

        caches['A' + str(l)] = A
        caches['Z' + str(l)] = activation_cache
        dropouts['D' + str(l)] = dropout

    Z = 0
    for k in range(0, L-1):
        A_rel = caches['A' + str(k)]
        Z_temp, linear_cache = forward_linear(A_rel, parameters['W' + str(k) + ',' + str(L-1)],
                                              parameters['b' + str(k) + ',' + str(L-1)],)
        Z += Z_temp

    A_temp, activation_cache = softmax(Z)
    A = A_temp

    caches['A' + str(L-1)] = A
    caches['Z' + str(L-1)] = activation_cache

    # assert (A.shape == (10, X.shape[1]))

    return A, caches, dropouts


def dense_dropout_backward_prop(AL, Y, parameters, caches, dropouts, keep_prob, layer_sizes):
    """ Implements backward propagation through all layers of ultra dense neural network
        Arguments:
            AL -- predicted output value of network
            Y -- label vector for supervised learning
            parameters -- dictionary of weight and bias values
            caches -- dictionary of Z and A values from forward propagation

        ReturnsP
        grads -- dictionary with the cost gradients with respect to A, W, and b
    """

    grads = {}
    param_grads = {}
    L = len(layer_sizes)
    m = Y.shape[0]
    epsilon = 1e-8

    # Initialize back prop for softmax
    grads['dA' + str(L - 1)] = - np.divide(Y, AL.T + epsilon)
    grads['dZ' + str(L - 1)] = softmax_backward(AL, Y)

    # Loop over layers and calculate gradients
    for l in reversed(range(0, L-1)):

        A = caches['A' + str(l)]
        dA_prev = np.zeros(caches['A' + str(l)].shape)

        for k in reversed(range(l+1, L)):
            # Parameter gradients
            dW_temp = (1 / m) * np.dot(grads['dZ' + str(k)], A.T)
            db_temp = (1 / m) * np.sum(grads['dZ' + str(k)], axis=1, keepdims=True)
            param_grads['dW' + str(l) + ',' + str(k)] = dW_temp
            param_grads['db' + str(l) + ',' + str(k)] = db_temp
            # Activation gradient
            if l != 0:
                dropout = dropouts['D' + str(l)]
                dA_temp = np.multiply(np.dot(parameters['W' + str(l) + ',' + str(k)].T, grads['dZ' + str(k)]),
                                      dropout)/keep_prob
                dA_prev += dA_temp

        # Store linear and activation gradient for next iteration
        if l != 0:
            grads['dA' + str(l)] = dA_prev
            grads['dZ' + str(l)] = leaky_relu_backward(grads['dA' + str(l)], caches['Z' + str(l)])

    return grads, param_grads


def update_parameters_with_adam(parameters, v, s, t, grads, learning_rate, layer_sizes,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Updates parameters using momentum.
    :param parameters: dictionary of parameter values
    :param v: dictionary containing current velocity values
    :param s: dictionary containing current RMSprop values
    :param t: sets bias correction iterator for momentum and RMSprop terms
    :param grads: dictionary of gradient values
    :param learning_rate: sets the learning rate for gradient descent with momentum
    :param layer_sizes: list containing the number of hidden units in each layer
    :param beta1: sets parameter value for the velocity based beta term
    :param beta2: sets parameter value for the RMSprop based beta term
    :param epsilon: sets additional constant value in S calculation for stability
    :return: parameters: updated dictionary of parameter values
             v: updated values of the current velocity
    """

    v_corrected = {}
    s_corrected = {}
    L = len(layer_sizes)

    for l in range(1, L):
        for k in range(l):
            v['dW' + str(k) + ',' + str(l)] = beta1 * v['dW' + str(k) + ',' + str(l)] + \
                                              (1 - beta1) * grads['dW' + str(k) + ',' + str(l)]
            v['db' + str(k) + ',' + str(l)] = beta1 * v['db' + str(k) + ',' + str(l)] + \
                                              (1 - beta1) * grads['db' + str(k) + ',' + str(l)]

            v_corrected['dW' + str(k) + ',' + str(l)] = v['dW' + str(k) + ',' + str(l)] / (1 - np.power(beta1, t))
            v_corrected['db' + str(k) + ',' + str(l)] = v['db' + str(k) + ',' + str(l)] / (1 - np.power(beta1, t))

            s['dW' + str(k) + ',' + str(l)] = beta2 * s['dW' + str(k) + ',' + str(l)] + \
                                              (1 - beta2) * np.power(grads['dW' + str(k) + ',' + str(l)], 2)
            s['db' + str(k) + ',' + str(l)] = beta2 * s['db' + str(k) + ',' + str(l)] + \
                                              (1 - beta2) * np.power(grads['db' + str(k) + ',' + str(l)], 2)

            s_corrected['dW' + str(k) + ',' + str(l)] = s['dW' + str(k) + ',' + str(l)] / (1 - np.power(beta2, t))
            s_corrected['db' + str(k) + ',' + str(l)] = s['db' + str(k) + ',' + str(l)] / (1 - np.power(beta2, t))

            parameters['W' + str(k) + ',' + str(l)] = \
                parameters['W' + str(k) + ',' + str(l)] - learning_rate * (v_corrected['dW' + str(k) + ',' + str(l)] /
                                                                           (np.sqrt(s_corrected['dW' + str(k) + ',' + str(l)])
                                                                            + epsilon))
            parameters['b' + str(k) + ',' + str(l)] = \
                parameters['b' + str(k) + ',' + str(l)] - learning_rate * (v_corrected['db' + str(k) + ',' + str(l)] /
                                                                           (np.sqrt(s_corrected['db' + str(k) + ',' + str(l)])
                                                                            + epsilon))

    return parameters, v, s


def compute_softmax_cost(AL, Y, epsilon):
    """
    Computes the cost of the softmax function after forward prop
    :param AL: The output matrix of the network of size (num examples, num classes)
    :param Y:  The label data matrix with the same size as AL
    :param epsilon: a small constant for numerical stability
    :return: cost: scalar value of cost
    """
    m = Y.shape[0]
    cost = (1 / m) * np.sum(np.sum(-np.multiply(Y, np.log(AL.T + epsilon)), axis=1, keepdims=True))

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


def predict_softmax(X, Y, parameters, layer_sizes):
    """ Uses the trained learned to predict an output from the neural network
        Arguments:
            parameters -- learned parameters
            X -- features used to predict output

        Returns:
            predictions -- predicted output from ultra dense neural network
    """

    Y_classes = np.reshape(np.argmax(Y, axis=1), (-1, 1))
    predictions, activations, dropouts = dense_dropout_forward_prop(X, parameters, 1, layer_sizes)

    predictions = np.reshape(np.argmax(predictions, axis=0), (-1, 1))

    num_correct = predictions[predictions == Y_classes].shape[0]
    accuracy = num_correct/Y.shape[0]

    return predictions, accuracy


def MNIST_model(X, Y, layer_sizes, learning_rate, num_epochs, mini_batch_size, alpha, sigma, beta1=0.9,
                beta2=0.999, epsilon=1e-8, keep_prob=1):
    """
    Constructs and trains a UDDNN for softmax classification of the MNIST data set
    :param X: input training data
    :param Y: input training labels for supervised learning
    :param X_val: input validation data - NOT CURRENTLY USED
    :param Y_val: input validation labels for supervised learning - NOT CURRENTLY USED
    :param layer_sizes: defines architecture of network, each element is the number of hidden units in that layer
    :param learning_rate: definition of alpha in gradient descent
    :param num_epochs: defines the number of epochs to iterate through in mini_batch gradient descent
    :param mini_batch_size: defines the mini_batch size
    :param epsilon: sets a small constant added in various functions for numeric stability
    :return: parameters: dictionary of parameter values after training
    """
    # Create validation and training sets
    np.random.seed(0)
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffled = X[permutation, :]
    Y_shuffled = Y[permutation, :]
    X = X_shuffled[0:50000, :]
    Y = Y_shuffled[0:50000, :]
    X_val = X_shuffled[50000:60000, :]
    Y_val = Y_shuffled[50000:60000, :]

    # Initialize variables
    t = 0
    parameters = initialize_dense_parameters(layer_sizes)
    v, s = initialize_dense_adam(layer_sizes)

    # Transpose input data for training
    #X_val = X_val.T

    print('Network is thinking')
    for i in tqdm(range(0, num_epochs)):

        cost = 0
        val_cost = 0
        # test_cost = 0
        X_train = np.empty((X.shape[1], X.shape[0]))
        Y_train = np.empty((Y.shape[0], Y.shape[1]))

        # Create mini-batches
        # Learning rate schedule
        if i == 1 and mini_batch_size < 2048:
            mini_batch_size = 2048
        elif (i + 1) % 75 == 0 and mini_batch_size >= 2048:
            learning_rate = learning_rate * 0.3
        mini_batches = create_mini_batches(X.T, Y, mini_batch_size, i+1)
        mini_batch_num = 0

        for batch in mini_batches:
            # Define each batch of X and Y from mini_batches
            (X_batch, Y_batch) = batch

            # Data augmentation, randomized elastic transformations and store deformed images in variable
            X_deformed = di.deform_images(X_batch.T, 28, alpha, sigma, i+1)
            X_deformed = X_deformed.T
            X_train[:, mini_batch_num * mini_batch_size:(mini_batch_num+1) * mini_batch_size] = X_batch
            Y_train[mini_batch_num * mini_batch_size:(mini_batch_num + 1) * mini_batch_size, :] = Y_batch

            # Forward propagation of inputs with dropout
            AL, caches, dropouts = dense_dropout_forward_prop(X_deformed, parameters, keep_prob, layer_sizes)
            mini_batch_num += 1

            # Perform back propagation with dropout
            grads, param_grads = dense_dropout_backward_prop(AL, Y_batch, parameters, caches, dropouts, keep_prob,
                                                             layer_sizes)

            # Update parameters with adam
            t += 1
            parameters, v, s = update_parameters_with_adam(parameters, v, s, t, param_grads, learning_rate,
                                                           layer_sizes, beta1, beta2, epsilon)

        # Save weights dictionary to file
        run.save()
        with open('UDNN-sweep-Weights/' + run.name + '.pkl', 'wb') as f:
            pickle.dump(parameters, f, 0)

        #####################################################################################
        #################################### METRICS ########################################

        # Compute cost of train, validation, and test set and test set after each epoch
        AL, caches, dropouts = dense_dropout_forward_prop(X_train, parameters, 1, layer_sizes)
        cost += compute_softmax_cost(AL, Y_train, epsilon)
        print('Average training cost after epoch %i: %f' % (i + 1, cost))

        val_AL, val_caches, dropouts = dense_dropout_forward_prop(X_val.T, parameters, 1, layer_sizes)  # Validation set output
        val_cost += compute_softmax_cost(val_AL, Y_val, epsilon)  # Validation set cost
        print('Average validation cost after epoch %i: %f' % (i + 1, val_cost))

        #test_AL, test_caches = ud.dense_forward_prop(X_val, parameters, layer_sizes)  # Test set output
        #test_cost += ud.compute_softmax_cost(test_AL, Y_val, epsilon)  # Test set cost
        #print('Average test cost after epoch %i: %f' % (i + 1, test_cost))

        # Train, validation, and test accuracy
        predictions, accuracy = predict_softmax(X_train, Y_train, parameters, layer_sizes)
        val_predictions, val_accuracy = predict_softmax(X_val.T, Y_val, parameters, layer_sizes)
        print('Training set accuracy: %f' % accuracy)
        print('Validation set accuracy: %f' % val_accuracy)

        # Create vector of parameter gradients for histogram
        weight_grads = np.array([[0]])
        for key in param_grads:
            weight_grads = np.concatenate((weight_grads, np.reshape(param_grads[key], (-1, 1))), axis=0)

        # Create vector of parameter values for histogram
        weight_vec = np.array([[0]])
        for key in parameters:
            weight_vec = np.concatenate((weight_vec, np.reshape(parameters[key], (-1, 1))), axis=0)

        # Log metrics in Weights and Biases
        wandb.log({'Training Loss': cost, 'Validation Loss': val_cost,
                   'Training Accuracy': accuracy, 'Validation Accuracy': val_accuracy,
                   'Gradient Evolution': wandb.Histogram(weight_grads, num_bins=512),
                   'Parameter Evolution': wandb.Histogram(weight_vec, num_bins=512)})

        ##################################### END METRICS ####################################
        ######################################################################################

    # test_predictions, test_accuracy = ud.predict_softmax(X_val, Y_val, parameters, layer_sizes)

    # Log hyper-parameters in Weights and Biases
    #wandb.log({'Architecture': layer_sizes, 'Number of Epochs': num_epochs, 'Learning Rate': learning_rate,
    #          'Mini-Batch Size': mini_batch_size, 'alpha': alpha, 'sigma': sigma})  # 'Test Accuracy': test_accuracy})

    # Finish logging to Weights and Biases
    wandb.join()

    return parameters, caches, grads


def main():
    X_train, Y_train, X_test, Y_test = imprep.get_mnist_data((28, 28))

    MNIST_model(
        X=X_train,
        Y=Y_train,
        layer_sizes=config.layer_sizes,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        mini_batch_size=config.mini_batch_size,
        alpha=config.alpha,
        sigma=config.sigma,
        keep_prob=config.keep_prob
        )


if __name__ == '__main__':
    main()
