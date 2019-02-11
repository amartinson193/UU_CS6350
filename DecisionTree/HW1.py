"""
Author: John Jacobson (u1201441)
Submission: 2/10/2019

This is an implementation of the ID3 algorithm for CS6350 at University of Utah in Spring 2019.
Supports Entropy, Majority Error, and Gini index gain.
Also supports arbitrary categorical attributes, with a single label per example.

Apologies for several cases of lazy implementation. There are numerous opportunities for improving runtime.

Should make this work for multiple labels, but not necessary for HW1


"""

import collections
import scipy.stats as stats
import numpy
import math


########################################################################################################
##########                              BEGIN BUILD TREE                                      ##########
########################################################################################################
########################################################################################################

def majority_error(labels):
    """
    This function returns majority error for a set of labels, in a discrete distribution.
    :param labels: List of labels, whose labels are frequency of each label.
    :return: Majority error of this label set.
    """
    count = sum(labels.values())
    majority_label = get_majority_label(labels)
    result = 0
    for value in labels:
        if value == majority_label:
            continue
        result += labels[value]
    return result / count


def gini_index(labels):
    """
    This function returns Gini Index for a set of labels, in a discrete distribution.
    :param labels: Dictionary of labels, whose labels are frequency of each label.
    :return: Gini Index of this label set.
    """
    count = sum(labels.values())
    gi = 1
    for value in labels:
        gi -= (labels[value] / count)**2
    return gi


def entropy(labels):
    """
    This function returns entropy for a set of labels, in a discrete distribution.
    :param labels: Dictionary of labels, whose labels are frequency of each label.
    :return: Entropy of this label set.
    """
    count = sum(labels.values())
    entropy = 0
    for value in labels:
        if labels[value] == 0:
            continue
        else:
            frequency = labels[value] / count
        entropy -= frequency * numpy.log(frequency)
    return entropy


def get_majority_label(attribute):
    majority = 0
    for value in attribute:
        if attribute[value] > majority:
            majority_element = value
            majority = attribute[value]
    return majority_element


def get_attribute_values(examples, attribute_index):
    """
    Get distinct values from all examples of a given attribute
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :return: Dictionary of distinct values within the given attribute, where dict values are frequency of each value.
    """
    values = {}
    for instance in examples:
        if instance[attribute_index] in values:
            values[instance[attribute_index]] += 1
        else:
            values[instance[attribute_index]] = 1
    return values


def get_attribute_values_numeric(examples, attribute_index, median):
    """
    Get counts of examples with given attribute above or not above the median of the whole set.
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :param median: Median of values in given attribute.
    :return: Dictionary with keys 1, -1 representing 1: greater than median, -1: less than median,
        value is frequency of each scenario
    """
    values = {}
    for instance in examples:
        if instance[attribute_index] > median:
            if 1 in values:
                values[1] += 1
            else:
                values[1] = 1
        else:
            if -1 in values:
                values[-1] += 1
            else:
                values[-1] = 1
    return values


def get_attribute_with_label(examples, attribute_index):
    """
    Get all value, label pairs from a set of examples.
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :return: List of pair list, each containing an example value and related label.
    """
    examples_trans = numpy.array(examples).transpose().tolist()
    attributes = examples_trans[attribute_index]
    labels = examples_trans[len(examples_trans) - 1]
    return list(zip(attributes, labels))


def create_label_list(examples, attribute_index, value):
    """
    Get labels and frequency tied to a specific value within an attribute.
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :param value: Specific value within attribute.
    :return: Dict of labels, where value is frequency of each label.
    """
    labels = {}
    label_index = len(examples[0]) - 1
    for instance in examples:
        if instance[attribute_index] == value:
            if instance[label_index] in labels:
                labels[instance[label_index]] += 1
            else:
                labels[instance[label_index]] = 1
    return labels


def create_label_list_numeric(examples, attribute_index, value, median):
    """
    Get labels and frequency tied to elements either greater than or less than the median of an attribute.
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :param value: Specific value within attribute.
    :param median: Median of the given attribute within the provided example set.
    :return: Dict of labels, where value is frequency of each label.
    """
    labels = {}
    label_index = len(examples[0]) - 1
    for instance in examples:
        if value == 1:
            if instance[attribute_index] > median:
                if instance[label_index] in labels:
                    labels[instance[label_index]] += 1
                else:
                    labels[instance[label_index]] = 1
        else:
            if instance[attribute_index] <= median:
                if instance[label_index] in labels:
                    labels[instance[label_index]] += 1
                else:
                    labels[instance[label_index]] = 1

    return labels


def get_median(examples, col):
    """
    Returns the median of one column from examples.
    :param examples: List of examples, each of which is a list of values.
    :param col: Index of column to extract median from.
    :return: Median of one column of all provided examples as a float.
    """
    values = []
    for instance in examples:
        values.append(instance[col])
    return numpy.median(values)


def get_key_by_max_value(dictionary):
    """
    Return the key of a dictionary with the largest value of all keys.
    :param dictionary: Input dictionary
    :return: Key of provided dictionary.
    """
    max = [0, None]
    for key in dictionary:
        if dictionary[key] > max[0]:
            max = [dictionary[key], key]
    return max[1]


def information_gain(examples, attribute_index, info_gain_type):
    """
    Calculates the potential gain if a set of examples were to be split by a specific attribute.
    May calculate using entropy, majority error, or Gini index.
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :param info_gain_type: integer to identify preferred method of gain.
        1 - Entropy
        2 - Majority Error
        3 - Gini Index
    :return: float
    """
    if info_gain_type == 1 :
        purity_func = entropy
    elif info_gain_type == 2:
        purity_func = majority_error
    elif info_gain_type == 3:
        purity_func = gini_index

    labels = get_attribute_values(examples, len(examples[0]) - 1)
    gain = purity_func(labels)
    values = get_attribute_values(examples, attribute_index)

    for value in values:
        value_labels = create_label_list(examples, attribute_index, value)
        gain -= purity_func(value_labels) * sum(value_labels.values()) / len(examples)

    return gain


def information_gain_numeric(examples, attribute_index, info_gain_type):
    """
    Calculates the potential gain if a set of examples were to be split by a specific attribute.
    May calculate using entropy, majority error, or Gini index.
    This function is used for numeric attributes only. Will convert to binary attributes using Median of examples.
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :param info_gain_type: integer to identify preferred method of gain.
        1 - Entropy
        2 - Majority Error
        3 - Gini Index
    :return: float
    """
    if info_gain_type == 1:
        purity_func = entropy
    elif info_gain_type == 2:
        purity_func = majority_error
    elif info_gain_type == 3:
        purity_func = gini_index

    median = get_median(examples, attribute_index)

    # need to add logic for numeric labels, but not necessary for assignment.
    labels = get_attribute_values(examples, len(examples[0]) - 1)
    gain = purity_func(labels)
    values = get_attribute_values_numeric(examples, attribute_index, median)

    for value in values:
        value_labels = create_label_list_numeric(examples, attribute_index, value, median)
        gain -= purity_func(value_labels) * sum(value_labels.values()) / len(examples)

    return gain


def get_next_attribute(examples, attributes, info_gain_type, numeric_cols):
    """
    Calculates the potential gain if a set of examples were to be split by a specific attribute.
    May calculate using entropy, majority error, or Gini index.
    This function is used for numeric attributes only. Will convert to binary attributes using Median of examples.
    :param examples: List of examples, each of which is a list of values.
    :param attributes: List of all attributes available to split.
    :param info_gain_type: integer to identify preferred method of gain.
        1 - Entropy
        2 - Majority Error
        3 - Gini Index
    :param numeric_cols: List of columns which are numeric, must be identified when passing initial dataset.
    :return: index of attribute with highest gain.
    """
    next_attribute = (-1,-1)
    for attribute in attributes:
        if attribute in numeric_cols:
            gain = information_gain_numeric(examples, attribute, info_gain_type)
        else:
            gain = information_gain(examples, attribute, info_gain_type)
        if gain > next_attribute[1]:
            next_attribute = (attribute, gain)
    return next_attribute[0]


def get_examples_by_value(examples, attribute_index, value):
    """
    Creates a list of examples containing the given value within the given attribute.
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :param value: Value within this attribute to identify desired examples
    :return: List of examples, each of which is a list of values, where the attribute_index
        element is equal to the value parameter.
    """
    example_subset = []
    for instance in examples:
        if instance[attribute_index] == value:
            example_subset.append(instance)
    return example_subset


def get_examples_by_value_numeric(examples, attribute_index, value, median):
    """
    Creates a list of examples containing examples either greater than median if value is 1, or not if value is -1.
    :param examples: List of examples, each of which is a list of values.
    :param attribute_index: Index of attribute to extract within examples.
    :param value: Value within this attribute to identify desired examples
    :param median: Median of attribute_index column of provided examples.
    :param value: Value within this attribute to identify desired examples
    :return: List of examples, each of which is a list of values, where the attribute_index
        element is greater than the median or not depending on value parameter.
    """
    example_subset = []
    for instance in examples:
        if value == 1:
            if instance[attribute_index] > median:
                example_subset.append(instance)
        else:
            if instance[attribute_index] <= median:
                example_subset.append(instance)
    return example_subset


def id3(examples, attributes, labels, max_depth, info_gain_type, numeric_cols):
    """
    Recursive ID3 implementation.
    :param examples: List of examples, each of which is a list of values.
    :param attributes: List of attribute indices.
    :param labels: Dictionary of labels, where values are frequencies of each label.
    :param max_depth: Maximum depth to grow from this node.
    :param info_gain_type:  integer to identify preferred method of gain.
        1 - Entropy
        2 - Majority Error
        3 - Gini Index
    :param numeric_cols: List of indices of numeric columns, to be provided with initial data.
    :return: node containing either an attribute to split, or a label to assign.
    """

    # Only one label in remaining data, return leaf node with this label.
    if len(labels.keys()) == 1:
        return list(labels)[0]
    # If no more attributes to split, return most common label
    if len(attributes) == 0 or max_depth == 0:
        return get_key_by_max_value(labels)

    next_attribute_numeric = False
    # Recursive step, create a root node
    node = tree()
    # Choose splitting attribute; store index of attribute in math.inf key. Branches will be stored in keys containing
    # their names, and will have tree values to subtrees/leaves.
    # 1 - entropy
    # 2 - majority error
    # 3 - gini index
    node[math.inf] = get_next_attribute(examples,attributes, info_gain_type, numeric_cols)
    if node[math.inf] in numeric_cols:
        next_attribute_numeric = True
    node[-math.inf] = get_key_by_max_value(labels) # add most common label in case unknown attribute values found
    new_attributes = list(attributes)
    new_attributes.remove(node[math.inf])

    # lazily handling numeric values with separate functions.
    # for numeric values, index 0 contains reference value (median), while -1 is a branch for values less than median,
    # 1 is branch for those greater than median.
    if next_attribute_numeric:
        node[0] = get_median(examples, node[math.inf])
        examples_less = get_examples_by_value_numeric(examples, node[math.inf], -1, node[0])
        if len(examples_less) == 0:
            return get_key_by_max_value(labels)
        # Otherwise, recursively add the next subtree
        new_labels = get_attribute_values(examples_less, len(examples[0]) - 1)
        node[-1] = id3(examples_less, new_attributes, new_labels, max_depth - 1, info_gain_type, numeric_cols)

        examples_greater = get_examples_by_value_numeric(examples, node[math.inf], 1, node[0])
        if len(examples_greater) == 0:
            return get_key_by_max_value(labels)
        # Otherwise, recursively add the next subtree
        new_labels = get_attribute_values(examples_greater, len(examples[0]) - 1)
        node[1] = id3(examples_greater, new_attributes, new_labels, max_depth - 1, info_gain_type, numeric_cols)
    else:
        # Iterate through values v of a (not label, but value of the attribute, like tall or short for height)
        values = get_attribute_values(examples, node[math.inf])
        for value in values:
            # Add a branch for v
            # Choose new e_v, examples with value v for attribute a
            examples_v = get_examples_by_value(examples, node[math.inf], value)
            # If e_v empty, add some result leaf to this branch; for now, use most common label in examples
            if len(examples_v) == 0:
                return get_key_by_max_value(labels)

            # Otherwise, recursively add the next subtree
            new_labels = get_attribute_values(examples_v, len(examples[0])-1)
            node[value] = id3(examples_v, new_attributes, new_labels, max_depth - 1, info_gain_type, numeric_cols)

    return node


def data_parsing(csv_file):
    """
    Reads in a file to a list of lists.
    This code was provided in the assignment description and used directly.
    :param csv_file: File to be read
    :return: List of lists where each member list is a comma separated line from file.
    """
    data = []
    with open(csv_file, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

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


def fill_missing_values(examples, missing_identifier):
    """
    Replaces missing values with the most common values in their attribute.
    NOTE: This modifies the data set passed into the algorithm directly; to work with unmodified data, it must be
        read in again.
    :param examples: List of examples, each of which is a list of values.
    :param missing_identifier: Data within examples indicating a missing value.
    :return: None
    """
    for index in range(len(examples[0])):
        attribute_dict = {}
        for instance in examples:
            if instance[index] in attribute_dict:
                attribute_dict[instance[index]] += 1
            else:
                attribute_dict[instance[index]] = 1
        if missing_identifier in attribute_dict:
            attribute_dict.pop(missing_identifier)
            new_value = get_key_by_max_value(attribute_dict)
            for instance_replace in examples:
                if instance_replace[index] == missing_identifier:
                    instance_replace[index] = new_value


def tree():
    """
    Implementation of a tree using defaultdict. Idea was found from
        https://stackoverflow.com/questions/3009935/looking-for-a-good-python-tree-data-structure
    sample usage -
    t = Tree()
    t[1][1][1][1] = 6
    print(t[1][1][1][1])
    :return: defaultdict of defaultdicts from collections.
    """
    return collections.defaultdict(tree)


def build_decision_tree(examples, max_depth, info_gain_type, numeric_cols, missing_identifier):

    labels = get_attribute_values(examples, len(examples[0])-1)
    if missing_identifier is not None:
        fill_missing_values(examples, missing_identifier)

    return id3(examples, list(range(len(examples[0])-1)), labels, max_depth, info_gain_type, numeric_cols)

########################################################################################################
##########                                BEGIN TEST TREE                                     ##########
########################################################################################################
########################################################################################################


def get_label(learned_tree, example, numeric_cols):
    """

    :param learned_tree: Learned tree
    :param example: List of examples, each of which is a list of values.
    :param numeric_cols: list of columns which are numeric.
    :return: Label that learned tree assigns the given example.
    """
    attribute_index = learned_tree[math.inf]

    # If this is a numeric attribute, compare the value to median (stored in tree[0]);
    # if greater, go to branch 1, else branch -1.
    if attribute_index in numeric_cols:
        value = float(example[attribute_index])
        if value > learned_tree[0]:
            lookup = learned_tree[1]
        else:
            lookup = learned_tree[-1]
    else:
        lookup = learned_tree[example[attribute_index]]

    if not isinstance(lookup, collections.defaultdict):
        return lookup
    if math.inf in lookup:
        return get_label(lookup, example, numeric_cols)
    else:
        return learned_tree[-math.inf]


def test_tree(learned_tree, examples, numeric_cols):
    """
    Tests data against a learned tree and reports error.
    :param learned_tree: Learned tree
    :param examples: List of examples, each of which is a list of values.
    :param numeric_cols: list of columns which are numeric.
    :return: integer number of matches, and integer number of total examples.
    """
    label_index = len(examples[0])-1
    actual_labels = [inst[label_index] for inst in examples]

    learned_labels = []
    for instance in examples:
        label = get_label(learned_tree, instance, numeric_cols)
        learned_labels.append(label)

    matches = 0
    for i in range(len(learned_labels)):
        if actual_labels[i] == learned_labels[i]:
            matches += 1
    return matches, len(actual_labels)



########################################################################################################
##########                                   BEGIN MAIN                                       ##########
########################################################################################################
########################################################################################################


def test():
    """
    Main method for learning and testing decision tree, and displaying results.
    :return: None
    """

    # test on car data set
    data = data_parsing("car/train.csv")
    test_data = data_parsing("car/test.csv")
    numeric_cols = []
    missing_identifier = None
    # info gain type - 1 for entropy, 2 for majority error, 3 for gini index.
    for gain in range(1,4):
        for depth in range(1,8):
            learned_tree = build_decision_tree(data, depth, gain, numeric_cols, missing_identifier)
            error_nums = test_tree(learned_tree, test_data, numeric_cols)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Using", gain_type, "and a maximum tree depth of", depth,
                  ",", error_nums[0], "of", error_nums[1], "examples were correctly mapped,",
                  "demonstrating an error rate of", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    # test on bank data set
    data = data_parsing("bank/train.csv")
    test_data = data_parsing("bank/test.csv")
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    map_numeric_data(data,numeric_cols) # convert numeric data to int type (for this specific application)
    missing_identifier = None
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = build_decision_tree(data, depth, gain, numeric_cols, missing_identifier)
            error_nums = test_tree(learned_tree, test_data, numeric_cols)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Using", gain_type, "and a maximum tree depth of", depth,
                  ",", error_nums[0], "of", error_nums[1], "examples were correctly mapped,",
                  "demonstrating an error rate of", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    # test on bank data set, with unknown values.
    data = data_parsing("bank/train.csv")
    test_data = data_parsing("bank/test.csv")
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    map_numeric_data(data,numeric_cols) # convert numeric data to int type (for this specific application)
    missing_identifier = "unknown"
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = build_decision_tree(data, depth, gain, numeric_cols, missing_identifier)
            error_nums = test_tree(learned_tree, test_data, numeric_cols)
            missing_identifier = None
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Using", gain_type, "and a maximum tree depth of", depth,
                  ",", error_nums[0], "of", error_nums[1], "examples were correctly mapped,",
                  "demonstrating an error rate of", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))


test()
