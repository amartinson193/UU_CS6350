# Created by John Jacobson (u1201441)
# Should make this work for multiple labels, but not necessary for HW1


import collections
import scipy.stats as stats
import numpy
import math


########################################################################################################
##########                              BEGIN BUILD TREE                                      ##########
########################################################################################################
########################################################################################################

# TODO
# Calculate Majority Error of given attribute
def majority_error(labels):
    """
    This function returns information gain for a set of examples if it were split on a provided attribute.
    Gain here is calculated using majority error (for a discrete distribution.)
    Higher information gain implies a better split.
    :param examples: List of all examples
    :param labels: Attribute being considered for next split
    :return: Information Gain if this set were split on provided attribute.
    """
    count = sum(labels.values())
    majority_label = get_majority_label(labels)
    result = 0
    for value in labels:
        if value == majority_label:
            continue
        result += labels[value]
    return result / count



# TODO
def gini_index(labels):
    """
    This function returns information gain for a set of examples if it were split on a provided attribute.
    Gain here is calculated using the Gini Index (for a discrete distribution.)
    Higher information gain implies a better split.
    :param examples: List of all examples
    :param attribute: Attribute being considered for next split
    :return: Information Gain if this set were split on provided attribute.
    """
    count = sum(labels.values())
    gi = 1
    for value in labels:
        gi -= (labels[value] / count)**2
    return gi


# pass in label/value pairs
def entropy(labels):
    count = sum(labels.values())
    entropy = 0
    for value in labels:
        if labels[value] == 0:
            continue
        else:
            frequency = labels[value] / count
        entropy -= frequency * numpy.log2(frequency)
    return entropy


def get_majority_label(attribute):
    majority = 0
    for value in attribute:
        if attribute[value] > majority:
            majority_element = value
            majority = attribute[value]
    return majority_element


# returns one 'column' of attributes or labels,presuming examples are fed as rows.
def get_attribute_values(examples, attribute_index):
    # examples_trans = numpy.array(examples).transpose().tolist()
    # return examples_trans[len(examples_trans)-1]
    values = {}
    for instance in examples:
        if instance[attribute_index] in values:
            values[instance[attribute_index]] += 1
        else:
            values[instance[attribute_index]] = 1
    return values


# returns one 'column' of attributes or labels,presuming examples are fed as rows.
def get_attribute_values_numeric(examples, attribute_index, median):
    # examples_trans = numpy.array(examples).transpose().tolist()
    # return examples_trans[len(examples_trans)-1]
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


# takes in full list of example lists, and an index, and returns the set of value,label pairs for all instances
# provided in examples
def get_attribute_with_label(examples, attribute_index):
    examples_trans = numpy.array(examples).transpose().tolist()
    attributes = examples_trans[attribute_index]
    labels = examples_trans[len(examples_trans) - 1]
    return list(zip(attributes, labels))


# pass in [value,label] pairs, this should peel off and return all labels.
def create_label_list(examples, attribute_index, value):
    labels = {}
    label_index = len(examples[0]) - 1
    for instance in examples:
        if instance[attribute_index] == value:
            if instance[label_index] in labels:
                labels[instance[label_index]] += 1
            else:
                labels[instance[label_index]] = 1
    return labels


# pass in [value,label] pairs, this should peel off and return all labels.
def create_label_list_numeric(examples, attribute_index, value, median):
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
    values = []
    for instance in examples:
        values.append(instance[col])
    return numpy.median(values)


# should receive full list of [example] lists, including all attributes and labels. Also the index of the attribute
# to calculate gain on.
# Don't need the full gain value for the algorithm, could use max purity, but nice to see the true gain calcs.
def information_gain(examples, attribute_index, info_gain_type):
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


# should receive full list of [example] lists, including all attributes and labels. Also the index of the attribute
# to calculate gain on.
# Don't need the full gain value for the algorithm, could use max purity, but nice to see the true gain calcs.
def information_gain_numeric(examples, attribute_index, info_gain_type, numeric_cols):
    if info_gain_type == 1 :
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


# TODO
# Calculate next best attribute to split on.
# should probably return index of attribute within examples.
def get_next_attribute(examples, attributes, info_gain_type, numeric_cols):
    next_attribute = (-1,-1)
    for attribute in attributes:
        if attribute in numeric_cols:
            gain = information_gain_numeric(examples, attribute, info_gain_type, numeric_cols)
        else:
            gain = information_gain(examples, attribute, info_gain_type)
        if gain > next_attribute[1]:
            next_attribute = (attribute, gain)
    return next_attribute[0]


def get_examples_by_value(examples, attribute_index, value):
    example_subset = []
    for instance in examples:
        if instance[attribute_index] == value:
            example_subset.append(instance)
    return example_subset


def get_examples_by_value_numeric(examples, attribute_index, value, median):
    example_subset = []
    for instance in examples:
        if value == 1:
            if instance[attribute_index] > median:
                example_subset.append(instance)
        else:
            if instance[attribute_index] <= median:
                example_subset.append(instance)
    return example_subset


# TODO
# Recursively build our decision tree.
#
# Currently returning most common label of entire set if a new value is found, but would be better to use the most
# common label at every leaf. Could do this by adding a negative inf index at every leaf addition with current most
# common label, possibly.
#
# labels - {label: count}
# examples - [[ex1],[ex2],...]
# attributes - {attribute: count}
# max_depth - int
def id3(examples, attributes, labels, max_depth, info_gain_type, numeric_cols):
    """Recursive ID3 learning algorithm"""

    # Only one label in remaining data, return leaf node with this label.
    if len(labels.keys()) == 1:
        return list(labels)[0]
    # If no more attributes to split, return most common label
    if len(attributes) == 0 or max_depth == 0:
        index = numpy.argmax(labels.values())
        return list(labels)[index]

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
    index = numpy.argmax(labels.values())
    node[-math.inf] = list(labels)[index] # add most common label in case unknown attribute values found
    new_attributes = list(attributes)
    new_attributes.remove(node[math.inf])
    remaining_attributes = attributes
    remaining_attributes.remove(node[math.inf])

    # lazily handling numeric values with separate functions.
    # for numeric values, index 0 contains reference value (median), while -1 is a branch for values less than median,
    # 1 is branch for those greater than median.
    if next_attribute_numeric:
        node[0] = get_median(examples, node[math.inf])
        examples_less = get_examples_by_value_numeric(examples, node[math.inf], -1, node[0])
        if len(examples_less) == 0:
            index = numpy.argmax(labels.values())
            return list(labels)[index]
        # Otherwise, recursively add the next subtree
        new_labels = get_attribute_values(examples_less, len(examples[0]) - 1)
        node[-1] = id3(examples_less, new_attributes, new_labels, max_depth - 1, info_gain_type, numeric_cols)

        examples_greater = get_examples_by_value_numeric(examples, node[math.inf], 1, node[0])
        if len(examples_greater) == 0:
            index = numpy.argmax(labels.values())
            return list(labels)[index]
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
                index = numpy.argmax(labels.values())
                return list(labels)[index]

            # Otherwise, recursively add the next subtree
            new_labels = get_attribute_values(examples_v, len(examples[0])-1)
            node[value] = id3(examples_v, new_attributes, new_labels, max_depth - 1, info_gain_type, numeric_cols)

    return node


# From assignment description
def data_parsing(csv_file):
    """ parse a file and read contents to a list """
    data = []
    with open(csv_file, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

    return data


# Convert numerical data - should improve this, but lazy for now.
def map_numeric_data(data, col_nums):
    for example in data:
        for col in col_nums:
            example[col] = int(example[col])


# From https://stackoverflow.com/questions/3009935/looking-for-a-good-python-tree-data-structure
# t = Tree()
# t[1][1][1][1] = 6
# print(t[1][1][1][1])
def tree():
    return collections.defaultdict(tree)


# TODO
def build_decision_tree(examples, max_depth, info_gain_type, numeric_cols):
    labels = get_attribute_values(examples, len(examples[0])-1)
    return id3(examples, list(range(len(examples[0])-1)), labels, max_depth, info_gain_type, numeric_cols)


########################################################################################################
##########                                BEGIN TEST TREE                                     ##########
########################################################################################################
########################################################################################################

def get_label(tree, example, numeric_cols):
    attribute_index = tree[math.inf]

    if attribute_index in numeric_cols:
        value = float(example[attribute_index])
        if value > float(tree[0]):
            lookup = tree[1]
        else:
            lookup = tree[-1]
    else:
        lookup = tree[example[attribute_index]]

    if not isinstance(lookup, collections.defaultdict):
        return lookup
    if math.inf in lookup:
        return get_label(lookup, example, numeric_cols)
    else:
        return tree[-math.inf]


def test_tree(tree, examples, numeric_cols):
    label_index = len(examples[0])-1
    actual_labels = [inst[label_index] for inst in examples]

    learned_labels = []
    for instance in examples:
        label = get_label(tree,instance, numeric_cols)
        learned_labels.append(label)

    matches = 0
    for i in range(len(learned_labels)):
        if actual_labels[i] == learned_labels[i]:
            matches += 1
    return (matches, len(actual_labels))



########################################################################################################
##########                                   BEGIN MAIN                                       ##########
########################################################################################################
########################################################################################################


def test():
    # TESTS
    # root directory in visual Studio - DecisionTree
    # in pycharm - one level higher.

    # data = data_parsing("DecisionTree/BooleanClassifier_TrainingData.csv")
    # data.pop(0)


    # test on car data set
    data = data_parsing("DecisionTree/car/train.csv")
    test_data = data_parsing("DecisionTree/car/test.csv")
    numeric_cols = []
    # info gain type - 1 for entropy, 2 for majority error, 3 for gini index.
    for gain in range(1,4):
        for depth in range(1,8):
            learned_tree = build_decision_tree(data, depth, gain, numeric_cols)
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
    data = data_parsing("DecisionTree/bank/train.csv")
    test_data = data_parsing("DecisionTree/bank/test.csv")
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    map_numeric_data(data,numeric_cols) # convert numeric data to int type (for this specific application)
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = build_decision_tree(data, depth, gain, numeric_cols)
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
            if gain == 3 and depth == 16:
                print(learned_tree)

test()