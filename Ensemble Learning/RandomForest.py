"""
Author: John Jacobson (u1201441)
Created: 2019-02-17

This is an implementation of a random forest algorithm for CS6350 at University of Utah in Spring 2019.

    Coming soon

"""

import ID3
import random


LABEL_INDEX = -1
WEIGHT_INDEX = -1
INFO_GAIN_TYPE = 1


def bagged_trees(examples, iterations, sample_size, numeric_cols, missing_identifier):
    global LABEL_INDEX
    global WEIGHT_INDEX

    LABEL_INDEX = len(examples[0]) - 2
    WEIGHT_INDEX = len(examples[0]) - 1

    trees = []
    samples = []

    for t in range(iterations):
        samples.append(resample(examples, sample_size))
        tree = ID3.build_random_tree(samples[t], -1, INFO_GAIN_TYPE, numeric_cols, missing_identifier)
        results = ID3.test_tree(tree, examples, numeric_cols, missing_identifier)
        error = 1 - (results[0] / results[1])
        trees.append(tuple([tree, results]))

    return trees


def resample(examples, sample_size):
    samples = []

    for i in range(sample_size):
        next = random.randint(0,len(examples)-1)
        samples.append(examples[next])

    return samples


def get_label(hypothesis, example, numeric_cols, missing_identifier):
    guess = 0
    result = []

    for operand in hypothesis:
        label = ID3.get_label(operand[0], example, numeric_cols, missing_identifier)

        if label == "yes":
            guess += 1
        else:
            guess -= 1

        if guess < 0:
            result.append("no")
        else:
            result.append("yes")

    return result


def test_bagged_tree_hypothesis(hypothesis, example_param, numeric_cols, missing_identifier):

    if isinstance(example_param, str):
        examples = ID3.data_parsing(example_param, numeric_cols)
    elif isinstance(example_param, list):
        examples = example_param
    else:
        raise AttributeError("Invalid data type: Please pass either file path or list of examples to build tree.")

    actual_labels = [tuple([inst[LABEL_INDEX], inst[WEIGHT_INDEX]]) for inst in examples]

    total = 0
    for instance in examples:
        total += instance[WEIGHT_INDEX]

    results = []
    learned_labels = []

    for instance in examples:
        labels = get_label(hypothesis, instance, numeric_cols, missing_identifier)
        learned_labels.append(tuple([labels, instance[WEIGHT_INDEX]]))
        # learned_labels contains a tuple for every example,
        # and every tuple contains a list of labels, one per iteration

    for t in range(len(hypothesis)):  # For every iteration
        matches = 0
        for i in range(len(learned_labels)):  # For every example
            if actual_labels[i][0] == learned_labels[i][0][t]:
                matches += learned_labels[i][1]
        results.append(tuple([matches, total]))

    return results
