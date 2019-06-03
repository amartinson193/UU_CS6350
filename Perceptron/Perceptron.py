"""
Author: John Jacobson (u1201441)
Created: 2019-03-02

This is an implementation of the Perceptron algorithm, including standard, voted, and averaged output.


"""
import numpy
import IOUtilities


# Forgot to include augmentation to vectors! Need to add 1 to the end of all examples to account for translation (bias).
def perceptron(file_path, epochs, missing_identifier):

    data = IOUtilities.data_parsing_numeric(file_path)
    examples, labels, label_map = IOUtilities.data_to_array(data)

    weight = numpy.zeros(len(examples[0, :]))
    weights = [label_map]
    learning_rate = 1 / (10**3)
    correct_count = 0

    for t in range(epochs):
        for i in range(len(examples)):
            predictor = numpy.transpose(weight) @ examples[i, :]
            prediction = sgn(predictor)
            if prediction != labels[i]:
                weights.append(tuple([weight, correct_count]))
                weight = numpy.array(weight + learning_rate * labels[i] * examples[i, :])
                correct_count = 0
            else:
                correct_count += 1
    # Append final weight
    weights.append(tuple([weight, correct_count]))

    return weights


def sgn(x):
    if x < 0:
        return -1
    else:
        return 1


def get_label(perceptron, example, missing_identifier, perceptron_type):
    """

    :param perceptron:
    :param example:
    :param perceptron_type: Calculation method
        1 - Standard Perceptron
        2 - Voted Perceptron
        3 - Averaged Perceptron
    :return:
    """

    label_map = perceptron.pop(0)

    if perceptron_type == 1:
        predictor = numpy.transpose(numpy.array(perceptron[-1][0]))
        return sgn(predictor @ example)
    elif perceptron_type == 2:
        result = 0
        for predictor in perceptron:
            weight = numpy.transpose(predictor[0])
            result += predictor[1] * (weight @ example)
        return sgn(result)
    elif perceptron_type == 3:
        average = numpy.zeros(len(perceptron[0][0]))
        for predictor in perceptron:
            average += predictor[0]
        result = numpy.transpose(average) @ example

        if label_map[0] == sgn(result):
            return label_map[0]
        else:
            return label_map[1]


def test_perceptron(perceptron, test_file_path, missing_identifier, perceptron_type):

    data = IOUtilities.data_parsing_numeric(test_file_path)
    examples, labels, label_map = IOUtilities.data_to_array(data)

    predictions = []

    for example in examples:
        predictions.append(get_label(list(perceptron), example, missing_identifier, perceptron_type))

    correct = 0
    for i in range(len(labels)):
        if labels[i] * predictions[i] > 0:
            correct += 1

    return tuple([correct, len(labels)])
