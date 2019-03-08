"""
Author: John Jacobson (u1201441)
Created: 2019-02-17

This is an implementation of the Least Mean Squares (LMS) algorithm for CS6350 at University of Utah in Spring 2019.

    Coming soon

"""

import GradientDescent

LABEL_INDEX = -1
AUGMENT_INDEX = -1


def least_mean_squares(example_param, batch_size, iterations, learning_constant):

    if isinstance(example_param, str):
        examples = data_parsing(example_param)
    elif isinstance(example_param, list):
        examples = example_param
    else:
        raise AttributeError("Invalid data type: Please pass either file path or list of examples to build tree.")

    labels = [ex[LABEL_INDEX] for ex in examples]
    x = [ex[0:len(ex)-1] for ex in examples]

    return GradientDescent.gradient_descent(x, labels, [0]*(len(x[0])), batch_size, iterations, learning_constant)


def data_parsing(csv_file):
    """
    Reads in a file to a list of lists.
    This code was provided in the assignment description and used directly.
    :param csv_file: File to be read
    :param numeric_cols: List of indices indicating which columns are numeric
    :return: List of lists where each member list is a comma separated line from file.
    """
    global LABEL_INDEX
    global AUGMENT_INDEX
    data = []

    with open(csv_file, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
            data[-1].insert(len(data[-1])-1, '1')  # Augment with 1

    # Storing a weight factor in the last index.
    LABEL_INDEX = len(data[0])-1
    AUGMENT_INDEX = len(data[0])-2

    # convert numeric data to int type (for this specific application)
    map_numeric_data(data, list(range(len(data[0]))))

    return data


def map_numeric_data(data, col_nums):
    """
    Converts strings to floats in the provided column.
    NOTE: This modifies the data set passed into the algorithm directly; to work with unmodified data, it must be
        read in again.
    :param data: data to be modified
    :param col_nums: index of column to be modified
    :return: None
    """
    for example in data:
        for col in col_nums:
            example[col] = float(example[col])


# Must pass in example augmented with 1 in last position
def get_label(weight, example):
    return sum(x*y for x,y in zip(weight, example))


def test_lms(hypothesis, example_param):

    if isinstance(example_param, str):
        examples = data_parsing(example_param)
    elif isinstance(example_param, list):
        examples = example_param
    else:
        raise AttributeError("Invalid data type: Please pass either file path or list of examples to build tree.")

    labels = [ex[LABEL_INDEX] for ex in examples]
    predicted_labels = [get_label(hypothesis, ex[0:len(ex)-1]) for ex in examples]

    loss = 0

    for i in range(len(labels)):
        loss += 0.5 * (labels[i] - predicted_labels[i])**2

    return loss

