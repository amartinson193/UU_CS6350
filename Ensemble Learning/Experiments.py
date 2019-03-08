"""
Author: John Jacobson (u1201441)
Created: 2019-02-17

This is set of experimentation functions for comparing and testing various learning algorithms.

    Coming soon

"""
import numpy
import random

import ID3
import AdaBoost
import BaggedTrees
import RandomForest
import LeastMeanSquares
import GraphUtility
import IOUtilities
import Perceptron

# FILE_PATH = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/BooleanClassifier_TrainingData.csv"
FILE_PATH_TRAIN = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/bank/train.csv"
FILE_PATH_TEST = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/bank/test.csv"
# FILE_PATH_TRAIN = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/BooleanClassifier_TrainingData.csv"
# FILE_PATH_TEST = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/BooleanClassifier_TrainingData.csv"

numeric_cols = [] # [0, 5, 9, 11, 12, 13, 14]  # Bank data numeric cols
missing_identifier = None



########################################################################################################
##########                               ID3 Experiments                                      ##########
########################################################################################################
########################################################################################################

FILE_PATH = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/"


def id3_experiment():
    """
    Main method for learning and testing decision tree, and displaying results.
    :return: None
    """

    #### test on car data set
    numeric_cols = []
    missing_identifier = None
    # test tree against original training examples.
    for gain in range(1,4):
        for depth in range(1,7):
            learned_tree = ID3.build_decision_tree(FILE_PATH + "car/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, FILE_PATH + "car/train.csv", numeric_cols, missing_identifier)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Car Data; Training; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))
    # test tree against test examples.
    for gain in range(1,4):
        for depth in range(1,7):
            learned_tree = ID3.build_decision_tree(FILE_PATH + "car/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, FILE_PATH + "car/test.csv", numeric_cols, missing_identifier)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Car data; Test; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    #### test on bank data set against training examples, no missing values
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    missing_identifier = None
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = ID3.build_decision_tree(FILE_PATH + "bank/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, FILE_PATH + "bank/train.csv", numeric_cols, missing_identifier)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Bank data, no missing data; Training; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    # test on bank data set against test examples.
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    missing_identifier = None
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = ID3.build_decision_tree(FILE_PATH + "bank/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, FILE_PATH + "bank/test.csv", numeric_cols, missing_identifier)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Bank data, no missing data; Test; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    #### test on bank data set, with unknown values.
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    missing_identifier = "unknown"
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = ID3.build_decision_tree(FILE_PATH + "bank/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, FILE_PATH + "bank/train.csv", numeric_cols, missing_identifier)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Bank data with missing data; Training; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    # test on bank data set, with unknown values.
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    missing_identifier = "unknown"
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = ID3.build_decision_tree(FILE_PATH + "bank/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, FILE_PATH + "bank/test.csv", numeric_cols, missing_identifier)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Bank data with missing data; Test; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))


########################################################################################################
##########                             AdaBoost Experiments                                   ##########
########################################################################################################
########################################################################################################

def ada_boost_experiment():
    examples = ID3.data_parsing(FILE_PATH_TRAIN, numeric_cols)

    # hypothesis = AdaBoost.ada_boost(examples, 5, numeric_cols, missing_identifier)

    # print(hypothesis)

    iterations = 100

    hypothesis = AdaBoost.ada_boost(examples, iterations, numeric_cols, missing_identifier)

    ada_results_train = AdaBoost.test_ada_boost_hypothesis(hypothesis, FILE_PATH_TRAIN, numeric_cols, missing_identifier)
    ada_results_test = AdaBoost.test_ada_boost_hypothesis(hypothesis, FILE_PATH_TEST, numeric_cols, missing_identifier)

    # for t in range(iterations):
    #     print("AdaBoost Training Set - t:", t, "results:", ada_results_train[t],
    #           "{0:.2%}".format(1-ada_results_train[t][0]/ada_results_train[t][1]))
    # for t in range(iterations):
    #     print("AdaBoost Testing Set - t:", t, "results:", ada_results_test[t],
    #           "{0:.2%}".format(1-ada_results_test[t][0]/ada_results_test[t][1]))
    # for t in range(iterations):
    #     tree_results = ID3.test_tree(hypothesis[t][0],FILE_PATH_TRAIN, numeric_cols, missing_identifier)
    #     print("Decision Tree Training Set - t:", t, "results:", tree_results,
    #           "{0:.2%}".format(1 - tree_results[0] / tree_results[1]))
    # for t in range(iterations):
    #     tree_results = ID3.test_tree(hypothesis[t][0],FILE_PATH_TEST, numeric_cols, missing_identifier)
    #     print("Decision Tree Test Set - t:", t, "results:", tree_results,
    #           "{0:.2%}".format(1 - tree_results[0] / tree_results[1]))

    ada_train = []
    ada_test = []
    dec_train = []
    dec_test = []

    for t in range(iterations):
        ada_train.append(1-ada_results_train[t][0]/ada_results_train[t][1])
        ada_test.append(1-ada_results_test[t][0]/ada_results_test[t][1])
        tree_results = ID3.test_tree(hypothesis[t][0], FILE_PATH_TRAIN, numeric_cols, missing_identifier)
        dec_train.append(1-tree_results[0]/tree_results[1])
        tree_results = ID3.test_tree(hypothesis[t][0],FILE_PATH_TEST, numeric_cols, missing_identifier)
        dec_test.append(1-tree_results[0]/tree_results[1])

    ada_graph = [tuple([ada_train, "AdaBoost Train"]), tuple([ada_test, "AdaBoost Test"])]
    GraphUtility.graph(ada_graph, "AdaBoost Data", "Iterations", "Error")

    tree_graph = [tuple([dec_train, "Tree Train"]), tuple([dec_test, "Tree Test"])]
    GraphUtility.graph(tree_graph, "Decision Tree Data", "Iterations", "Error")


########################################################################################################
##########                             Bagging Experiments                                    ##########
########################################################################################################
########################################################################################################

def bagged_trees_experiment():
    examples = ID3.data_parsing(FILE_PATH_TRAIN, numeric_cols)

    # hypothesis = AdaBoost.ada_boost(examples, 5, numeric_cols, missing_identifier)

    # print(hypothesis)

    LABEL_INDEX = len(examples[0])-2

    iterations = 100
    sample_size = int(len(examples) / 2)

    hypothesis = BaggedTrees.bagged_trees(examples, iterations, sample_size, numeric_cols, missing_identifier)
    results_train = BaggedTrees.test_bagged_tree_hypothesis(hypothesis, FILE_PATH_TRAIN, numeric_cols, missing_identifier)
    results_test = BaggedTrees.test_bagged_tree_hypothesis(hypothesis, FILE_PATH_TEST, numeric_cols, missing_identifier)
    # for t in range(iterations):
    #     print("Bagged Tree Training Set - t:", t, "results:", results_train[t],
    #           "{0:.2%}".format(1-results_train[t][0]/results_train[t][1]))
    # for t in range(iterations):
    #     print("Bagged Tree Test Set - t:", t, "results:", results_test[t],
    #           "{0:.2%}".format(1-results_test[t][0]/results_test[t][1]))

    # Charts
    bag_train = []
    bag_test = []

    for t in range(iterations):
        bag_train.append(1-results_train[t][0]/results_train[t][1])
        bag_test.append(1-results_test[t][0]/results_test[t][1])

    bag_graph = [tuple([bag_train, "Bagging Train"]), tuple([bag_test, "Bagging Test"])]
    GraphUtility.graph(bag_graph, "Bagged Tree Data", "Num Trees", "Error")


    # Bias/Variance calculations.
    iterations = 100
    bagged_trees = []
    trees = []
    sample_size = 100

    for t in range(iterations):
        data = random.sample(examples, 1000)
        bagged_trees.append(BaggedTrees.bagged_trees(data, iterations, sample_size, numeric_cols, missing_identifier))
        trees.append(bagged_trees[t][0][0])

    # Bias/Variance of individual trees.
    labels = []
    biases = []
    variances = []
    for instance in examples:
        for tree in trees:
            label = ID3.get_label(tree, instance, numeric_cols, missing_identifier)
            labels.append(1 if label == "yes" else -1)

        if instance[LABEL_INDEX] == "yes":
            true_label = 1
        else:
            true_label = -1

        avg = numpy.average(labels)
        biases.append((avg - true_label)**2)
        variances.append(numpy.var(labels))

    tree_bias = numpy.average(biases)
    tree_variance = numpy.average(variances)

    # Bias/Variance of bagged trees.
    labels = []
    biases = []
    variances = []
    for instance in examples:
        for tree in bagged_trees:
            label = BaggedTrees.get_label(tree, instance, numeric_cols, missing_identifier)
            labels.append(1 if label == "yes" else -1)

        true_label = 1 if instance[LABEL_INDEX] == "yes" else -1

        avg = numpy.average(labels)
        biases.append((avg - true_label)**2)
        variances.append(numpy.var(labels))

    bag_bias = numpy.average(biases)
    bag_variance = numpy.average(variances)

    print("Tree Bias:", "{0:.3}".format(tree_bias), "Tree Variance:", "{0:.3}".format(tree_variance))
    print("Bagged Bias:", "{0:.3}".format(bag_bias), "Bagged Variance:", "{0:.3}".format(bag_variance))


########################################################################################################
##########                          Random Forest Experiments                                 ##########
########################################################################################################
########################################################################################################

def random_forest_experiment():
    examples = ID3.data_parsing(FILE_PATH_TRAIN, numeric_cols)

    # hypothesis = AdaBoost.ada_boost(examples, 5, numeric_cols, missing_identifier)

    # print(hypothesis)

    LABEL_INDEX = len(examples[0])-2

    iterations = 100
    sample_size = int(len(examples) / 4)

    for feature_size in [2,4,6]:
        hypothesis = RandomForest.random_forest(examples, iterations, sample_size, numeric_cols, missing_identifier, feature_size)
        results_train = RandomForest.test_random_forest_hypothesis(hypothesis, FILE_PATH_TRAIN, numeric_cols, missing_identifier)
        results_test = RandomForest.test_random_forest_hypothesis(hypothesis, FILE_PATH_TEST, numeric_cols, missing_identifier)

        # Charts
        forest_train = []
        forest_test = []
        for t in range(iterations):
            # print("Random Forest -", "Feature Size:", feature_size, "t:", t, "results:", results[t], results[t][0]/results[t][1])
            forest_train.append(1-results_train[t][0]/results_train[t][1])
            forest_test.append(1-results_test[t][0]/results_test[t][1])

        forest_graph = [tuple([forest_train, "Forest Train - " + str(feature_size) + " features"]),
                        tuple([forest_test, "Forest Test - " + str(feature_size) + " features"])]
        GraphUtility.graph(forest_graph, "Random Forest Data", "Num Trees", "Error")

    # Bias/Variance
    iterations = 100
    forest = []
    trees = []
    sample_size = 100
    feature_size = 2

    for t in range(iterations):
        data = random.sample(examples, 1000)
        forest.append(RandomForest.random_forest(data, iterations, sample_size, numeric_cols, missing_identifier, feature_size))
        trees.append(forest[t][0][0])

    # Bias/Variance of individual trees.
    labels = []
    biases = []
    variances = []
    for instance in examples:
        for tree in trees:
            label = ID3.get_label(tree, instance, numeric_cols, missing_identifier)
            labels.append(1 if label == "yes" else -1)

        if instance[LABEL_INDEX] == "yes":
            true_label = 1
        else:
            true_label = -1

        avg = numpy.average(labels)
        biases.append((avg - true_label)**2)
        variances.append(numpy.var(labels))

    tree_bias = numpy.average(biases)
    tree_variance = numpy.average(variances)

    # Bias/Variance of bagged trees.
    labels = []
    biases = []
    variances = []
    for instance in examples:
        for tree in forest:
            label = RandomForest.get_label(tree, instance, numeric_cols, missing_identifier)
            labels.append(1 if label == "yes" else -1)

        true_label = 1 if instance[LABEL_INDEX] == "yes" else -1

        avg = numpy.average(labels)
        biases.append((avg - true_label)**2)
        variances.append(numpy.var(labels))

    forest_bias = numpy.average(biases)
    forest_variance = numpy.average(variances)

    print("Tree Bias:", "{0:.3}".format(tree_bias), "Tree Variance:", "{0:.3}".format(tree_variance))
    print("Forest Bias:", "{0:.3}".format(forest_bias), "Forest Variance:", "{0:.3}".format(forest_variance))


########################################################################################################
##########                               LMS Experiments                                      ##########
########################################################################################################
########################################################################################################

def lms_experiment():

    file_path_train = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW2/concrete/train.csv"
    file_path_test = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW2/concrete/test.csv"

    batch_size = -1  # -1 for full batch descent, or a batch size for stochastic descent.
    iterations = 1000
    learning_constant = 1

    plot_iter = 1000

    results = []

    for t in range(plot_iter):
        hypothesis = LeastMeanSquares.least_mean_squares(file_path_train, batch_size, iterations, learning_constant)
        if hypothesis[2]:
            break
        learning_constant = learning_constant/2

    r = hypothesis[0]
    hypotheses = hypothesis[1]

    for hyp in hypotheses:
        results.append(LeastMeanSquares.test_lms(hyp, file_path_test))

    print("Gradient Descent - r:", r, "Weight:", hypotheses[-1],"Losses:", results)

    batch_size = 1  # -1 for full batch descent, or a small batch size for stochastic descent.
    results_stoch = []

    for t in range(plot_iter):
        hypothesis = LeastMeanSquares.least_mean_squares(file_path_train, batch_size, iterations, learning_constant)
        if hypothesis[2]:
            break
        learning_constant = learning_constant/2

    r = hypothesis[0]
    hypotheses = hypothesis[1]

    for hyp in hypotheses:
        results_stoch.append(LeastMeanSquares.test_lms(hyp, file_path_test))

    print("Stochastic Gradient Descent - r:", r, "Weight:", hypotheses[-1],"Losses:", results_stoch)

    lms_graph = [tuple([results, "Descent"]), tuple([results_stoch, "Stochastic Descent"])]

    GraphUtility.graph(lms_graph, "LMS_Test", "Loss", "Gradient Descent Iterations")


########################################################################################################
##########                             Credit Experiments                                     ##########
########################################################################################################
########################################################################################################

def credit_experiment():

    file_path = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW2/credit/default of credit card clients.csv"

    numeric_cols = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
    missing_identifier = None
    training_data = []
    test_data = []

    data = ID3.data_parsing(file_path, numeric_cols)

    LABEL_INDEX = len(data[0]) - 2

    for instance in data:
        if instance[LABEL_INDEX] == '1':
            instance[LABEL_INDEX] = "yes"
        else:
            instance[LABEL_INDEX] = "no"

    test_indices = random.sample(range(len(data)), len(data))
    for i in test_indices:
        if i < 6000:
            test_data.append(data[i])
        else:
            training_data.append(data[i])

    iterations = 100

    decision_tree = ID3.build_decision_tree(training_data, max_depth=-1, info_gain_type=1, numeric_cols=numeric_cols, missing_identifier=missing_identifier)
    adaboost = AdaBoost.ada_boost(training_data, iterations=iterations, numeric_cols=numeric_cols, missing_identifier=missing_identifier)
    bagged_tree = BaggedTrees.bagged_trees(training_data, iterations=iterations, sample_size=100, numeric_cols=numeric_cols, missing_identifier=missing_identifier)
    forest = RandomForest.random_forest(training_data, iterations=iterations, sample_size=100, numeric_cols=numeric_cols, missing_identifier=missing_identifier, feature_size=4)


    # Decision Tree results

    tree_results = ID3.test_tree(decision_tree, training_data, numeric_cols, missing_identifier)
    tree_train = 1-tree_results[0]/tree_results[1]
    tree_results = ID3.test_tree(decision_tree,test_data, numeric_cols, missing_identifier)
    tree_test = 1-tree_results[0]/tree_results[1]

    tree_train_ln = []
    tree_test_ln = []

    for t in range(iterations):
        tree_train_ln.append(tree_train)
        tree_test_ln.append(tree_test)

    # AdaBoost results
    ada_results_train = AdaBoost.test_ada_boost_hypothesis(adaboost, training_data, numeric_cols, missing_identifier)
    ada_results_test = AdaBoost.test_ada_boost_hypothesis(adaboost, test_data, numeric_cols, missing_identifier)

    ada_train = []
    ada_test = []

    for t in range(iterations):
        ada_train.append(1-ada_results_train[t][0]/ada_results_train[t][1])
        ada_test.append(1-ada_results_test[t][0]/ada_results_test[t][1])

    ada_graph = [tuple([ada_train, "AdaBoost Train"]), tuple([ada_test, "AdaBoost Test"]),
                 tuple([tree_train_ln, "Tree Train"]), tuple([tree_test_ln, "Tree Test"])]

    GraphUtility.graph(ada_graph, "AdaBoost Data", "Iterations", "Error")

    # Bagging results
    results_train = BaggedTrees.test_bagged_tree_hypothesis(bagged_tree, training_data, numeric_cols, missing_identifier)
    results_test = BaggedTrees.test_bagged_tree_hypothesis(bagged_tree, test_data, numeric_cols, missing_identifier)

    # Charts
    bag_train = []
    bag_test = []

    for t in range(iterations):
        bag_train.append(1-results_train[t][0]/results_train[t][1])
        bag_test.append(1-results_test[t][0]/results_test[t][1])

    bag_graph = [tuple([bag_train, "Bagging Train"]), tuple([bag_test, "Bagging Test"]),
                 tuple([tree_train_ln, "Tree Train"]), tuple([tree_test_ln, "Tree Test"])]
    GraphUtility.graph(bag_graph, "Bagged Tree Data", "Num Trees", "Error")

    # Forest Results
    results_train = RandomForest.test_random_forest_hypothesis(forest, training_data, numeric_cols, missing_identifier)
    results_test = RandomForest.test_random_forest_hypothesis(forest, test_data, numeric_cols, missing_identifier)

    # Charts
    forest_train = []
    forest_test = []
    for t in range(iterations):
        forest_train.append(1-results_train[t][0]/results_train[t][1])
        forest_test.append(1-results_test[t][0]/results_test[t][1])

    forest_graph = [tuple([forest_train, "Forest Train - " + str(2) + " features"]),
                    tuple([forest_test, "Forest Test - " + str(2) + " features"]),
                    tuple([tree_train_ln, "Tree Train"]), tuple([tree_test_ln, "Tree Test"])]
    GraphUtility.graph(forest_graph, "Random Forest Data", "Num Trees", "Error")


########################################################################################################
##########                           Manual LMS Regression                                    ##########
########################################################################################################
########################################################################################################

def manual_lms():

    file_path_train = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW2/concrete/train.csv"

    examples = LeastMeanSquares.data_parsing(file_path_train)

    labels = []

    for instance in examples:
        instance.pop(-1)
        labels.append(instance.pop(-1))

    ex = numpy.array(examples)
    x = ex.transpose()
    lbl = numpy.array(labels)
    y = lbl.reshape((53,1))

    xtrans = x.transpose()
    xprod = x @ xtrans
    xinv_x = numpy.linalg.inv(xprod) @ x
    weight = xinv_x @ y

    print(weight)


########################################################################################################
##########                                Perceptron                                          ##########
########################################################################################################
########################################################################################################


def perceptron_experiment():
    FILE_PATH_TRAIN = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW3/bank-note/bank-note/train.csv"
    FILE_PATH_TEST = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW3/bank-note/bank-note/test.csv"

    missing_identifier = None

    weights = Perceptron.perceptron(FILE_PATH_TRAIN, 10, missing_identifier)

    # Standard Perceptron Results
    print("Learned vector - ", weights[-1][0])
    # Training
    results = Perceptron.test_perceptron(weights, FILE_PATH_TRAIN,missing_identifier, 1)
    print("BankNote Data; Test; Standard Perceptron", "Correct -", results[0], "Total -",
          results[1], "Err -", "{0:.2%}".format(1-results[0]/results[1]))
    # Test
    results = Perceptron.test_perceptron(weights, FILE_PATH_TEST,missing_identifier, 1)
    print("BankNote Data; Test; Standard Perceptron", "Correct -", results[0], "Total -",
          results[1], "Err -", "{0:.2%}".format(1-results[0]/results[1]))


    # Voted Perceptron Results
    for i in range(1, len(weights)):
        print("Vector", i, "-", weights[i][0], "Correct Predictions -", weights[i][1])
    print("Total vectors -", len(weights)-1)
    # Training
    results = Perceptron.test_perceptron(weights, FILE_PATH_TRAIN,missing_identifier, 2)
    print("BankNote Data; Test; Voted Perceptron", "Correct -", results[0], "Total -",
          results[1], "Err -", "{0:.2%}".format(1-results[0]/results[1]))
    # Test
    results = Perceptron.test_perceptron(weights, FILE_PATH_TEST,missing_identifier, 2)
    print("BankNote Data; Test; Voted Perceptron", "Correct -", results[0], "Total -",
          results[1], "Err -", "{0:.2%}".format(1-results[0]/results[1]))



    # Average Perceptron Results

    average = numpy.zeros(len(weights[1][0]))
    for predictor in weights:
        if isinstance(predictor, dict):
            continue
        average += predictor[0]
    print("Learned vector - ", average)
    # Training
    results = Perceptron.test_perceptron(weights, FILE_PATH_TRAIN,missing_identifier, 3)
    print("BankNote Data; Test; Average Perceptron", "Correct -", results[0], "Total -",
          results[1], "Err -", "{0:.2%}".format(1-results[0]/results[1]))
    # Test
    results = Perceptron.test_perceptron(weights, FILE_PATH_TEST,missing_identifier, 3)
    print("BankNote Data; Test; Average Perceptron", "Correct -", results[0], "Total -",
          results[1], "Err -", "{0:.2%}".format(1-results[0]/results[1]))



