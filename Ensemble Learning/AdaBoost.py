"""
Author: John Jacobson (u1201441)
Created: 2019-02-17

This is an implementation of the AdaBoost algorithm for CS6350 at University of Utah in Spring 2019.

    Coming soon

"""
import math
import ID3


LABEL_INDEX = -1
WEIGHT_INDEX = -1


def ada_boost(examples, iterations, numeric_cols, missing_identifier):
    global LABEL_INDEX
    global WEIGHT_INDEX

    LABEL_INDEX = len(examples[0]) - 2
    WEIGHT_INDEX = len(examples[0]) - 1

    trees = []
    error = -1

    for t in range(iterations):
        if error == 0:
            break

        if t == 0:
            tree = None

        alpha = weight_examples(examples, tree, error, numeric_cols, missing_identifier)
        tree = ID3.build_decision_tree(examples, 1, 1, numeric_cols, missing_identifier)
        results = ID3.test_tree(tree, examples, numeric_cols, missing_identifier)
        error = 1 - (results[0] / results[1])
        trees.append(tuple([tree, alpha, error]))

    return trees


def weight_examples(examples, tree, error, numeric_cols, missing_identifier):
    if error == -1:
        num_examples = len(examples)
        for instance in examples:
            instance[WEIGHT_INDEX] = 1 / num_examples
        return 1

    weight_sum = 0
    alpha = -0.5 * math.log((1 - error) / error)

    # Calculate new weights.
    for instance in examples:
        true_label = instance[LABEL_INDEX]
        predicted_label = ID3.get_label(tree, instance, numeric_cols, missing_identifier)

        if predicted_label == true_label:
            sign = 1
        else:
            sign = -1

        instance[WEIGHT_INDEX] = instance[WEIGHT_INDEX] * math.exp(alpha * sign)
        weight_sum += instance[WEIGHT_INDEX]

    # Normalize the new weights.
    for instance in examples:
        instance[WEIGHT_INDEX] = instance[WEIGHT_INDEX] / weight_sum

    return alpha


def get_label(hypothesis, example, numeric_cols, missing_identifier):
    guess = 0
    result = []

    for operand in hypothesis:
        label = ID3.get_label(operand[0], example, numeric_cols, missing_identifier)

        if label == "yes":
            guess += operand[1]
        else:
            guess -= operand[1]

        if guess < 0:
            result.append("no")
        else:
            result.append("yes")

    return result


def test_ada_boost_hypothesis(hypothesis, example_param, numeric_cols, missing_identifier):

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
