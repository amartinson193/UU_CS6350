"""
Author: John Jacobson (u1201441)
Created: 2019-02-17

This is set of experimentation functions for comparing and testing various learning algorithms.

    Coming soon

"""

import ID3
import AdaBoost
import BaggedTrees
import RandomForest

# FILE_PATH = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/BooleanClassifier_TrainingData.csv"
FILE_PATH_TRAIN = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/bank/train.csv"
FILE_PATH_TEST = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/bank/test.csv"
# FILE_PATH_TRAIN = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/BooleanClassifier_TrainingData.csv"
# FILE_PATH_TEST = "/home/john/PycharmProjects/u1201441_Private_Repository/CS6350_Files/HW1/Data/BooleanClassifier_TrainingData.csv"

numeric_cols = [] # [0, 5, 9, 11, 12, 13, 14]  # Bank data numeric cols
missing_identifier = None


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
    results = AdaBoost.test_ada_boost_hypothesis(hypothesis, FILE_PATH_TEST, numeric_cols, missing_identifier)
    for t in range(iterations):
        print("AdaBoost - t:", t, "results:", results[t], results[t][0]/results[t][1])


########################################################################################################
##########                             Bagging Experiments                                    ##########
########################################################################################################
########################################################################################################

def bagged_trees_experiment():
    examples = ID3.data_parsing(FILE_PATH_TRAIN, numeric_cols)

    # hypothesis = AdaBoost.ada_boost(examples, 5, numeric_cols, missing_identifier)

    # print(hypothesis)

    iterations = 100
    sample_size = len(examples)

    hypothesis = BaggedTrees.bagged_trees(examples, iterations, sample_size, numeric_cols, missing_identifier)
    results = BaggedTrees.test_bagged_tree_hypothesis(hypothesis, FILE_PATH_TEST, numeric_cols, missing_identifier)
    for t in range(iterations):
        print("Bagged Tree - t:", t, "results:", results[t], results[t][0]/results[t][1])





########################################################################################################
##########                          Random Forest Experiments                                 ##########
########################################################################################################
########################################################################################################





########################################################################################################
##########                               LMS Experiments                                      ##########
########################################################################################################
########################################################################################################


