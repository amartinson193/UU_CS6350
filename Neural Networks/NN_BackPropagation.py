"""
Author: John Jacobson (u1201441)
Submission: 4/26/2019

This is an implementation of a dense multi-layer perceptron artificial neural network using backpropagation via stochastic
gradient descent.

"""
import numpy as np
import IOUtilities


def shuffle_in_unison(a, b):
    """
    Function for shuffling elements of 2 equal length lists to same state, i.e. indices of each list are shuffled to the
     same order.
    :param a:
    :param b:
    :return:
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def sigmoid(z):
    """
    Sigmoid activation function for neural net node.
    :param z: numeric input
    :return: sigmoid of input
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(s):
    """
    Derivative of sigmoid function for backpropagation. This function assumes that input is already in sigmoid form
    for purposes of this algorithm.
    :param s: sigmoid form of numeric input.
    :return: derivative of sigmoid centered at input to param s.
    """
    return s * (1 - s)


def forward(example, weights):
    """
    Forward pass through neural network, mapping the input to a specific output and returning the entire associated
    network of nodes associated with this input.
    :param example: numpy array of input to be fed through network
    :param weights: list of numpy arrays containing weights for each layer transition
    :return: list of calculated layers (numpy arrays) output from given input example, and list of cached products
    (numpy arrays) from forward operations, for use in backprop.
    """
    layers = [example]  # every layer of model, including initial examples and final prediction.
    forward_cache = []  # cached products for backprop, one per weight layer.
    bias = np.array([1])

    for i in range(len(weights)-1):
        next_layer = sigmoid(layers[i].dot(weights[i]))
        next_layer = np.concatenate((next_layer, bias))
        layers.append(np.array(next_layer))
        # Take outer product of this layers input with activation derivative of this layers output.
        # This cache will be used later in backprop. Just need input and output layers, and during
        # backprop we will use weight matrix products to acquire the remaining derivative pieces.
        # Note that this value will eventually be added to the existing weight of this layer, so should
        # result in the same dimensions of the current weight layer.
        forward_cache.append(np.outer(layers[i], sigmoid_prime(layers[-1][:-1])))

    prediction = np.float64(layers[-1].dot(weights[-1]))
    layers.append(prediction)

    return layers, forward_cache


# mult by label error, then by every weight along the path.
# cache already contains input and output layer values, just need weights
def back_prop(layers, forward_cache, weights, true_label, learning_rate):
    """
    Backpropagation phase, updates given set of layer-transition weights in-place using stochastic gradient descent.
    :param layers: Calculated layers from forward pass of network (returned by forward())
    :param forward_cache: Cached products from forward pass of network (returned by forward())
    :param weights: Current list of weight arrays
    :param true_label: True output of current example for calculating loss
    :param learning_rate: Learning constant for SGD, should be updated with an appropriate learning schedule.
    :return: N/A
    """
    hidden_layers = len(layers) - 2  # initial layer is input, final layer is prediction.
    layer_width = layers[-2].size - 1
    err = layers[-1] - true_label

    # get non-bias elements from final layer and start backprop cache, i.e. cut last row.
    last_layer = np.array([weights[-1][:-1]])
    # rows = tuple([last_layer] * (len(last_layer) + 1) )
    backprop_cache = learning_rate * err * last_layer  # learning rate here propagates to all weights.
    if len(backprop_cache.shape) > 2:
        backprop_cache = np.squeeze(backprop_cache, axis=2)

    # Update weights.
    # Final layer is forward cache * error
    # Next layers all involve forward cache * error * weight matrix of previous layer weights.
    # Building up cache of weight products as we traverse.
    # Final layer of weights is special case with no weights factored in, so calc first.
    last_hidden = (learning_rate * err * layers[-2]).transpose()
    weights[-1] = np.squeeze(weights[-1]) - last_hidden  # recall layers[-1] is prediction, so -2 is final hidden layer.

    for i in range(len(forward_cache)-1, -1, -1):
        cur_layer_weights = np.delete(weights[i], -1, 0)  # final row is bias, so remove?
        rows = np.repeat(backprop_cache, repeats=len(layers[i]), axis=0)
        cache = np.multiply(forward_cache[i], np.array(rows))
        weights[i] = weights[i] - cache
        if i > 0:
            backprop_cache = backprop_cache @ cur_layer_weights


# Generate weights from a Gaussian distribution
def gaussian_weights(example_len, hidden_layers, layer_width):
    weights = []

    # Weights for input to first hidden layer
    weights.append(np.random.normal( size=(example_len, layer_width)))

    # Weights from hidden layer to hidden layer
    layer_weight_size = (layer_width+1, layer_width)
    for i in range(hidden_layers-1):
        weights.append(np.random.normal( size=layer_weight_size))

    # Weights from final hidden layer to output
    weights.append(np.random.normal( size=(layer_width+1, 1)))

    return weights


# Generate weight arrays of zeros.
def zero_weights(example_len, hidden_layers, layer_width):
    weights = []

    # Weights for input to first hidden layer
    weights.append(np.zeros( shape=(example_len, layer_width)))

    # Weights from hidden layer to hidden layer
    layer_weight_size = (layer_width+1, layer_width)
    for i in range(hidden_layers-1):
        weights.append(np.zeros( shape=layer_weight_size))

    # Weights from final hidden layer to output
    weights.append(np.zeros( shape=(layer_width+1, 1)))

    return weights


def train_network(file_path, epochs, rate_schedule, layer_width, weight_init):

    data = IOUtilities.data_parsing_numeric(file_path)
    examples, labels, label_map = IOUtilities.data_to_array(data)

    if weight_init == "gaussian":
        weight_list = gaussian_weights(examples.shape[1], 2, layer_width)
    else:
        weight_list = zero_weights(examples.shape[1], 2, layer_width)

    learning_rate = 0.01  # gamma

    for t in range(epochs):
        shuffle_in_unison(examples, labels)
        for i in range(len(examples)):
            forward_result, cache = forward(examples[i, :], weight_list)
            back_prop(forward_result, cache, weight_list, labels[i], learning_rate)

        learning_rate = rate_schedule(learning_rate, t, d=2)

    return label_map, weight_list


def sgn(x):
    if x < 0:
        return -1
    else:
        return 1


def get_label(hypothesis, example, label_map):
    """

    :param hypothesis:
    :param example:
    :param label_map:
    :return:
    """

    layers, cache = forward(example, hypothesis)
    result = sgn(layers[-1])

    if label_map[0] == result:
        return label_map[0]
    else:
        return label_map[1]


def test_nn(hypothesis, test_file_path):

    data = IOUtilities.data_parsing_numeric(test_file_path)
    examples, labels, label_map = IOUtilities.data_to_array(data)

    predictions = []

    for example in examples:
        predictions.append(get_label(hypothesis, example, label_map))

    correct = 0
    for i in range(len(labels)):
        if labels[i] * predictions[i] > 0:
            correct += 1

    return correct, len(labels)






