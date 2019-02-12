
import ID3


########################################################################################################
##########                                   BEGIN MAIN                                       ##########
########################################################################################################
########################################################################################################


def test():
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
            learned_tree = ID3.build_decision_tree("car/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, "car/train.csv", numeric_cols, missing_identifier)
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
            learned_tree = ID3.build_decision_tree("car/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, "car/test.csv", numeric_cols, missing_identifier)
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
            learned_tree = ID3.build_decision_tree("bank/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, "bank/train.csv", numeric_cols, missing_identifier)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Bank data; Training; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    # test on bank data set against test examples.
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    missing_identifier = None
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = ID3.build_decision_tree("bank/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, "bank/test.csv", numeric_cols, missing_identifier)
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Bank data; Test; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    #### test on bank data set, with unknown values.
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    missing_identifier = "unknown"
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = ID3.build_decision_tree("bank/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, "bank/train.csv", numeric_cols, missing_identifier)
            missing_identifier = None
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Bank data; Training; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))

    # test on bank data set, with unknown values.
    numeric_cols = [0,5,9,11,12,13,14] # columns with numeric data
    missing_identifier = "unknown"
    for gain in range(1,4):
        for depth in range(1,17):
            learned_tree = ID3.build_decision_tree("bank/train.csv", depth, gain, numeric_cols, missing_identifier)
            error_nums = ID3.test_tree(learned_tree, "bank/test.csv", numeric_cols, missing_identifier)
            missing_identifier = None
            if gain == 1:
                gain_type = "Entropy"
            elif gain == 2:
                gain_type = "Majority Error"
            else:
                gain_type = "Gini Index"
            print("Bank data; Test; Gain -", gain_type, "Depth -", depth, "Correct -", error_nums[0], "Total -",
                  error_nums[1],"Err -", "{0:.2%}".format(1-error_nums[0]/error_nums[1]))


test()
