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
def majority_error(attribute):
    """
    This function returns information gain for a set of examples if it were split on a provided attribute.
    Gain here is calculated using majority error (for a discrete distribution.)
    Higher information gain implies a better split.
    :param examples: List of all examples
    :param attribute: Attribute being considered for next split
    :return: Information Gain if this set were split on provided attribute.
    """
    return 0



# TODO
def gini_index(attribute):
    """
    This function returns information gain for a set of examples if it were split on a provided attribute.
    Gain here is calculated using the Gini Index (for a discrete distribution.)
    Higher information gain implies a better split.
    :param examples: List of all examples
    :param attribute: Attribute being considered for next split
    :return: Information Gain if this set were split on provided attribute.
    """
    return 0


# pass in attribute dict with count of each attribute
def entropy(attribute):
    count = sum(attribute.values())
    result = 0
    for value in attribute:
        if attribute[value] == 0:
            continue
        else:
            frequency = attribute[value] / count
        result -= frequency * numpy.log2(frequency)
    return result


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


# should receive full list of [example] lists, including all attributes and labels. Also the index of the attribute
# to calculate gain on.
# Don't need the full gain value for the algorithm, could use max purity, but nice to see the true gain calcs.
def information_gain(examples, attribute_index, infoGainType):
    if infoGainType == 1 :
        purity_func = entropy
    elif infoGainType == 2:
        purity_func = majority_error
    elif infoGainType == 3:
        purity_func = gini_index

    labels = get_attribute_values(examples, len(examples[0]) - 1)
    gain = purity_func(labels)
    values = get_attribute_values(examples, attribute_index)

    for value in values:
        value_labels = create_label_list(examples, attribute_index, value)
        gain -= purity_func(value_labels) * sum(value_labels.values()) / len(examples)

    return gain



# TODO
# Calculate next best attribute to split on.
# should probably return index of attribute within examples.
def get_next_attribute(examples, attributes, infoGainType):
    next_attribute = (-1,-1)
    for attribute in attributes:
        gain = information_gain(examples, attribute, infoGainType)
        if gain > next_attribute[1]:
            next_attribute = (attribute, gain)
    return next_attribute[0]


def get_examples_by_value(examples, attribute_index, value):
    example_subset = []
    for instance in examples:
        if instance[attribute_index] == value:
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
def id3(examples, attributes, labels, max_depth):
    """Recursive ID3 learning algorithm"""

    # Only one label in remaining data, return leaf node with this label.
    if len(labels.keys()) == 1:
        return list(labels)[0]
    # If no more attributes to split, return most common label
    if len(attributes) == 0 or max_depth == 0:
        index = numpy.argmax(labels.values())
        return list(labels)[index]

    # Recursive step, create a root node
    node = tree()
    # Choose splitting attribute; store index of attribute in math.inf key. Branches will be stored in keys containing their names, and will have tree values to subtrees/leaves.
    # 1 - entropy
    # 2 - majority error
    # 3 - gini index
    node[math.inf] = get_next_attribute(examples,attributes, 1)
    index = numpy.argmax(labels.values())
    node[-math.inf] = list(labels)[index] # add most common label in case unknown attribute values found
    new_attributes = list(attributes)
    new_attributes.remove(node[math.inf])
    remaining_attributes = attributes
    remaining_attributes.remove(node[math.inf])
    # Iterate through values v of a (not label, but value of the attribute, like tall or short for height)
    values = get_attribute_values(examples, node[math.inf])

    for value in values:
        # Add a branch for v
        # Choose new e_v, examples with value v for attribute a
        examples_v = get_examples_by_value(examples, node[math.inf], value)
        # If e_v empty, add some result leaf to this branch; for now, use most common label in examples
        if len(examples_v) == 0:
            index = numpy.argmax(labels.values())
            return labels.keys()[index]

        # Otherwise, recursively add the next subtree
        new_labels = get_attribute_values(examples_v, len(examples[0])-1)
        node[value] = id3(examples_v, new_attributes, new_labels, max_depth - 1)

    return node


# From assignment description
def data_parsing(csv_file):
    """ parse a file and read contents to a list """
    data = []
    with open(csv_file, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

    return data


# From https://stackoverflow.com/questions/3009935/looking-for-a-good-python-tree-data-structure
#
# t = Tree()
# t[1][1][1][1] = 6
# print(t[1][1][1][1])
def tree():
    return collections.defaultdict(tree)


# TODO
def build_decision_tree(examples, max_depth):
    labels = get_attribute_values(examples, len(examples[0])-1)
    return id3(examples, list(range(len(examples[0])-1)), labels, max_depth)


########################################################################################################
##########                                BEGIN TEST TREE                                     ##########
########################################################################################################
########################################################################################################

def get_label(tree, example):
    attribute_index = tree[math.inf]
    lookup = tree[example[attribute_index]]
    if not isinstance(lookup, collections.defaultdict):
        return lookup
    if math.inf in lookup:
        return get_label(lookup, example)
    else:
        return tree[-math.inf]

def test_tree(tree, examples):
    label_index = len(examples[0])-1
    actual_labels = [inst[label_index] for inst in examples]

    learned_labels = []
    for instance in examples:
        label = get_label(tree,instance)
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

    data = data_parsing("DecisionTree/car/train.csv")
    print(data)
    # for i in range(len(data[0]) - 1):
    #     print(information_gain(data, i))

    for i in range(1,7):
        learned_tree = build_decision_tree(data, i)
        print(learned_tree)

        # print(get_label(learned_tree, ['low','vhigh','4','4','big','med']))
        # print(get_label(learned_tree, ['low','high','5more','4','med','high']))
        # print(get_label(learned_tree, ['low','high','2','2','med','high']))
        test_data = data_parsing("DecisionTree/car/test.csv")

        print(test_tree(learned_tree, test_data))






def main():
    print(0)


test()
