"""
Author: John Jacobson (u1201441)
Created: 2019-03-02

This is an implementation of the Perceptron algorithm for CS6350 at University of Utah in Spring 2019.

    Coming soon

"""
import numpy
import IOUtilities
from scipy.optimize import minimize as min
from scipy.spatial.distance import pdist, squareform


def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)


def primal_svm(file_path, epochs, rate_schedule, weight_constant):

    data = IOUtilities.data_parsing_numeric(file_path)
    examples, labels, label_map = IOUtilities.data_to_array(data)

    examples = numpy.delete(examples,len(examples[0, :])-1,1)

    weight = numpy.zeros(len(examples[0, :]))
    weights = [label_map]
    learning_rate = 0.9  # gamma
    correct_count = 0

    for t in range(epochs):
        shuffle_in_unison(examples, labels)
        for i in range(len(examples)):
            predictor = weight.dot(examples[i, :])
            if predictor * labels[i] <= 1:
                weight = numpy.array((1 - learning_rate) * weight + learning_rate * weight_constant * len(examples) * labels[i] * examples[i, :])  # update weight
            else:
                weight[:-1] = (1 - learning_rate) * weight[:-1]  # update weight, but not bias parameter.
        learning_rate = rate_schedule(learning_rate,t)
    # Append final weight
    weights.append(weight)

    return weights


def dual_objective_function(alphas, subgradient):
    result = -sum(alphas)

    result += alphas.dot(subgradient).dot(alphas)

    return 0.5 * result


def constraint(alphas, labels):
    return alphas.dot(labels)


def gaussian_kernel(xi, xj, gamma):
    result = numpy.exp(-numpy.linalg.norm(xi-xj)**2 / gamma)
    return result


def dual_svm(file_path, weight_constant, gamma):

    data = IOUtilities.data_parsing_numeric(file_path)
    examples, labels, label_map = IOUtilities.data_to_array(data)

    labels = numpy.array(labels)

    weight = numpy.zeros(len(examples[0, :-1]))
    weights = [label_map]
    learning_rate = 1 / (10**3)  # gamma
    correct_count = 0

    bnds = []
    for i in range(len(examples)):
        bnds.append((0, weight_constant))
    cnst = {'type': 'eq', 'fun': constraint, 'args': (labels,)}

    if gamma is not None:
        pairwise_dists = squareform(pdist(examples, 'euclidean'))
        gram_matrix = numpy.exp(-pairwise_dists ** 2 / gamma)  # got this idea for optimization from stack exchange user bayerj
        #for i in range(len(examples)):
        #    for j in range(len(examples)):
        #        gram_matrix[i][j] = gaussian_kernel(examples[i], examples[j], gamma) * labels[i] * labels[j]
    else:
        gram_matrix = (examples @ examples.T)

    subgradient = gram_matrix * numpy.tile(labels.T, (len(gram_matrix), 1)).T * numpy.tile(labels,
                                                                                           (len(gram_matrix), 1))
    objective_result = min(dual_objective_function, numpy.zeros(len(examples)), args=subgradient, method='SLSQP', bounds=bnds, constraints=cnst)

    dual_result = objective_result.x

    if gamma is None:
        weight = (dual_result * labels).dot(examples)
    else:
        for i in range(len(weight)):
            weight += dual_result[i] * labels[i] * gaussian_kernel(examples[i],numpy.zeros(len(examples[i])), gamma)

    positive_alphas = 0

    for i in range(len(examples)):
        if dual_result[i] > 0:
            positive_alphas += 1


    b = labels - (dual_result * labels).dot(gram_matrix)


    b = sum(b)/positive_alphas

    weights.append(numpy.insert(weight, len(weight), b))
    weights.append(positive_alphas)

    weights.append(examples)
    weights.append(dual_result)
    weights.append(labels)
    return weights


def sgn(x):
    if x < 0:
        return -1
    else:
        return 1


def get_label(hypothesis, example, label_map):
    """

    :param perceptron:
    :param example:
    :param perceptron_type: Calculation method
        1 - Standard Perceptron
        2 - Voted Perceptron
        3 - Averaged Perceptron
    :return:
    """

    result = sgn(hypothesis[1].dot(example))

    if label_map[0] == result:
        return label_map[0]
    else:
        return label_map[1]


def test_primal_svm(hypothesis, test_file_path):

    data = IOUtilities.data_parsing_numeric(test_file_path)
    examples, labels, label_map = IOUtilities.data_to_array(data)

    examples = numpy.delete(examples, len(examples[0, :])-1,1)

    predictions = []
    label_map = hypothesis[0]

    for example in examples:
        predictions.append(get_label(hypothesis, example, label_map))

    correct = 0
    for i in range(len(labels)):
        if labels[i] * predictions[i] > 0:
            correct += 1

    return tuple([correct, len(labels)])


def get_label_kernel(hypothesis, example, label_map, gamma):
    """

    :param perceptron:
    :param example:
    :param perceptron_type: Calculation method
        1 - Standard Perceptron
        2 - Voted Perceptron
        3 - Averaged Perceptron
    :return:
    """

    examples = hypothesis[-3]
    alphas = hypothesis[-2]
    labels = hypothesis[-1]
    result = 0

    for i in range(len(hypothesis[-2])):
        result += alphas[i] * labels[i] * gaussian_kernel(examples[i],example, gamma)

    if label_map[0] == sgn(result):
        return label_map[0]
    else:
        return label_map[1]


def test_dual_svm(hypothesis, test_file_path):

    data = IOUtilities.data_parsing_numeric(test_file_path)
    examples, labels, label_map = IOUtilities.data_to_array(data)

    examples = numpy.delete(examples, len(examples[0, :])-1,1)

    predictions = []
    label_map = hypothesis[0]

    for example in examples:
        predictions.append(get_label(hypothesis, example, label_map))

    correct = 0
    for i in range(len(labels)):
        if labels[i] * predictions[i] > 0:
            correct += 1

    return tuple([correct, len(labels)])
