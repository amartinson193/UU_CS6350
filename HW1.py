
allData = []
trainingData = []
testData = []

# This contains the tree learned from the test data.
learnedFunction = []

# TODO
# Calculate Entropy of given attribute
def Entropy(attribute):
    return 0

# TODO
# Calculate Majority Error of given attribute
def MajorityError(attribute):
    return 0

# TODO
# Calculate next best attribute to split on.
def GetNextAttribute(infoGainType):
    if infoGainType == 1:
        return 0
    elif infoGainType == 2:
        return 0

# TODO
# Recursively build our decision tree.
# Flag for describing Information Gain method
#   1 - Entropy
#   2 - Majority Error
def ID3(Examples, Attributes, Labels):
    """Recursive ID3 learning algorithm"""
    #Only one label in remaining data, return leaf node with this label.
    if len(Labels) == 1:
        return Labels[0]

    #If no more attributes to split, return most common label
    if len(Attributes == 0):
        return stats.mode(Examples)
    #Recursive step, create a root node
    branches = 0
    leaves = 0
    #Choose splitting attribute
    a = GetNextAttribute(1)
    #Iterate through values v of a (not label, but value of the attribute, like tall or short for height)
    #Add a branch for v
    #Choose new Ev, examples with value v for attribute a
    Ev = []
    newLabels = []
    #If Ev empty, add some result leaf to this branch; for now, use most common Label in Examples
    #Otherwise, recursively add the next subtree
    remainingAttributes = Attributes
    remainingAttributes.remove(a)
    ID3(Ev, remainingAttributes, newLabels)

def DataInput():
    return 0

# TODO
# From assignment description
def DataParsing(CSVFile):
    """ parse a file and read contents to a list """
    with open(CSVFile,'r') as f:
        for line in f:
            terms = line.strip().split(',')
            # process one training example

# From https://stackoverflow.com/questions/3009935/looking-for-a-good-python-tree-data-structure
#
# t = Tree()
# t[1][1][1][1] = 6
# print(t[1][1][1][1])
def Tree():
    return collections.defaultdict(Tree)

# TODO
def BuildDecisionTree():
    ID3([1],[1],[1])


import collections
import scipy.stats as stats

print(stats.mode([1,1,2]))

