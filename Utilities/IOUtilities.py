"""
Author: John Jacobson (u1201441)
Created: 2019-03-02

This file contains data I/O utilities, functions for reading, writing,
and processing data for the included machine learning algorithms.


"""
import numpy


def data_parsing(csv_file, numeric_cols):
    """
    Reads in a file to a list of lists.
    This code was provided in the assignment description and used directly.
    :param csv_file: File to be read
    :param numeric_cols: List of indices indicating which columns are numeric
    :return: List of lists where each member list is a comma separated line from file.
    """
    global LABEL_INDEX
    global WEIGHT_INDEX
    data = []

    with open(csv_file, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
            data[len(data)-1].append(1)  # Add weight of 1 to all examples.

    # Storing a weight factor in the last index.
    LABEL_INDEX = len(data[0])-2
    WEIGHT_INDEX = len(data[0])-1

    if len(numeric_cols) > 0:
        map_numeric_data(data, numeric_cols)  # convert numeric data to int type (for this specific application)

    return data


def data_parsing_numeric(csv_file):
    """
    Reads in a file to a list of lists. Assumes all data in the file is numeric.
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
    NOTE: This modifies the data set in place.
    :param data: data to be modified
    :param col_nums: index of column to be modified
    :return: None
    """
    for example in data:
        for col in col_nums:
            example[col] = float(example[col])


def data_to_array(data):
    """
    Converts list of lists data to numpy arrays, and splits last index of every example off into a label array. Also
    maps all labels to +/- 1, and includes a label map for original labels. Also augments examples with trailing 1.
    :param data: List of lists to be converted.
    :return: instances numpy array, labels numpy array, label_map {label: +-/-1} dict.
    """
    label_map = {}

    label_index = len(data[0])-1

    array_data = numpy.array(data)

    instances = array_data[:, :-1]
    labels = array_data[:, label_index]

    label_val = -1
    for label in labels:
        # Map each label to a +/- numeric value arbitrarily.
        if label not in label_map:
            label_map[label] = label_val
            # After first label is mapped to -1, next should be +1
            label_val += 2
        # Only working with binary classification, so stop after both labels mapped.
        if label_val > 1:
            break

    labels = [label_map[l] for l in labels]

    # For augmenting examples, but this is currently being done in data parsing function so deprecated.
    # examples = numpy.ones((instances.shape[0], instances.shape[1]+1))
    # examples[:, :-1] = instances

    return instances, labels, label_map
