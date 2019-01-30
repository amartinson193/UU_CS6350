
import collections
import scipy.stats as stats
import numpy

# TODO
# Calculate Majority Error of given attribute
def gain_majority_error(examples, attribute):
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
def gain_gini_index(examples, attribute):
    """
    This function returns information gain for a set of examples if it were split on a provided attribute.
    Gain here is calculated using the Gini Index (for a discrete distribution.)
    Higher information gain implies a better split.
    :param examples: List of all examples
    :param attribute: Attribute being considered for next split
    :return: Information Gain if this set were split on provided attribute.
    """
    return 0


# TODO
# Calculate next best attribute to split on.
def get_next_attribute(infoGainType):
    if infoGainType == 1:
        return 0
    elif infoGainType == 2:
        return 0


# TODO
# Recursively build our decision tree.
# Flag for describing Information Gain method
#   1 - Entropy
#   2 - Majority Error
def id3(examples, attributes, labels, max_depth):
    """Recursive ID3 learning algorithm"""

    # Only one label in remaining data, return leaf node with this label.
    if len(labels) == 1:
        return labels[0]
    # If no more attributes to split, return most common label
    if len(attributes == 0):
        return stats.mode(examples)

    if max_depth == 0:
        return 0 # need to determine final step here, if max_depth has been reached.

    # Recursive step, create a root node
    branches = 0
    leaves = 0
    # Choose splitting attribute
    a = get_next_attribute(1)
    # Iterate through values v of a (not label, but value of the attribute, like tall or short for height)
    # Add a branch for v
    # Choose new e_v, examples with value v for attribute a
    e_v = []
    new_labels = []
    # If e_v empty, add some result leaf to this branch; for now, use most common label in examples
    # Otherwise, recursively add the next subtree
    remaining_attributes = attributes
    remaining_attributes.remove(a)
    id3(e_v, remaining_attributes, new_labels, max_depth - 1)


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
def build_decision_tree():
    id3([1],[1],[1], -1)


########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################

def main():
    """ Main function """


"""
all_data = []
training_data = []
test_data = []
print(stats.mode([1,1,2])[0][0])

file = open("testfile.csv","w")

file.write("This is a test,and another test")


data = data_parsing("DecisionTree/car/train.csv")

data = numpy.array(data)

print(data)

print(data.transpose())
    
test_dict = {1: 5, 2: 6}
print(1 in test_dict)
print(5 in test_dict)



values = [0,0,0,1]
labels = [0,0,1,1]
pair = zip(values,labels)
pair = list(pair)

print(entropy(pair))
"""


# pass in [label] list
def entropy(labels):
    count = len(labels)
    label_counts = {}
    result = 0
    for instance in labels:
        if instance in label_counts:
            label_counts[instance] += 1
        else:
            label_counts[instance] = 1
    for label in label_counts:
        frequency = label_counts[label] / count
        result -= frequency * numpy.log2(frequency)
    return result


# should receive pairs of values and labels, return list containing one list per value,
# containing value-label pair lists.
def split_attribute(examples):
    result_list = []
    values = {}

    for instance in examples:
        if instance[0] in values:
            values[instance[0]].append(instance)
        else:
            values[instance[0]] = []
            values[instance[0]].append(instance)

    for pairs in values:
        result_list.append(values[pairs])

    return result_list


# returns one 'column' of attributes or labels,presuming examples are fed as rows.
def get_attribute(examples, attribute_index):
    examples_trans = numpy.array(examples).transpose().tolist()
    return examples_trans[len(examples_trans)-1]


# takes in full list of example lists, and an index, and returns the set of value,label pairs for all instances
# provided in examples
def get_attribute_with_label(examples, attribute_index):
    examples_trans = numpy.array(examples).transpose().tolist()
    attributes = examples_trans[attribute_index]
    labels = examples_trans[len(examples_trans)-1]
    return list(zip(attributes,labels))


# pass in [value,label] pairs, this should peel off and return all labels.
def create_label_list(examples):
    labels = []
    for instance in examples:
        labels.append(instance[1])
    return labels


# should receive full list of [example] lists, including all attributes and labels. Also the index of the attribute
# to calculate gain on.
def information_gain(examples, attribute_index):
    labels = get_attribute(examples, len(examples)-1)
    gain = entropy(labels)

    attribute_pairs = get_attribute_with_label(examples,attribute_index)
    values = split_attribute(attribute_pairs)

    for value in values:
        value_labels = create_label_list(value)
        gain -= entropy(value_labels) * len(value_labels) / len(examples)

    return gain


# TESTS
data = data_parsing("DecisionTree/BooleanClassifier_TrainingData.csv")

data.pop(0)

print(data)

for i in range(len(data[0])-1):
    print(information_gain(data,i))




