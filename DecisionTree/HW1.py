
import collections
import scipy.stats as stats
import numpy


# TODO
# Calculate Entropy of given attribute
def entropy(examples, attribute):
    label_col = len(examples[0])-1
    label_dict = {}
    for i in range(len(examples)):
        label_dict[examples[i][label_col]] += 1
    else:
        label_dict[examples[i][label_col]] = 1
    return 0


# TODO
# Calculate Majority Error of given attribute
def majority_error(examples, attribute):
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
    id3([1],[1],[1])


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
"""




















