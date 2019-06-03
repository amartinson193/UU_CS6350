This is a machine learning library developed by John Jacobson for CS5350/6350 in University of Utah.

Please be aware that creating this code was my first exposure to Python, as well as to these algorithms. All code was 
created solely by me with no assistance, aside from online documentation and course materials, in my free time. I 
learned a lot over the course of creating this library, but obviously am still far from perfect. As I have free time I 
will work to improve and add more functionality as a learning opportunity, so I'm always open to suggestions.


Source
~~~~~~

Code may be found at https://github.com/u1201441/UU_CS6350

~~~~~~

Decision Tree
-------------
-------------

ID3
-------

This is an implementation of the ID3 algorithm for CS6350 at University of Utah in Spring 2019. Supports Entropy,
     Majority Error, and Gini index gain. Also supports arbitrary categorical attributes, with a single label per 
     example. Missing values are compensated by including weighted copies of the example to average out the missing 
     value.

Build Tree
~~~~~~~~~~

build_decision_tree
    args:
        1. file_path: String containing file path.
        2. max_depth: integer for maximum depth of decision tree.
        3. info_gain_type: integer to determine method for calculating gain.
            1 - Entropy
            2 - Majority Error
            3 - Gini Index
        4. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        5. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        The root of a DefaultDict decision tree. Each node is represented by either a DefaultDict subtree, or a label
         from the dataset as a leaf.
        Contents of node:
            node[math.inf]: Index of attribute within the example
            node[-math.inf]: Contains the most common training label at this branch in case no further decision can 
            be made.
            node[None]: Contains the most common training value for the attribute in math.inf in case the test 
            example is missing data or a new value on this attribute.
            node['value']: For some 'value' of the attribute in math.inf, this will either contain a subtree splitting on
             the next attribute or a leaf containing a label.

~~~~~~~~~~

Get Label
~~~~~~~~~

get_label
    args:
        1. learned_tree: A DefaultDict decision tree as returned by build_decision_tree function.
        2. example: A single training example as a list of values with a single label in the last index.
        3. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        4. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        A single label from the training set which the given tree assigns this example.

~~~~~~~~~

Test Tree
~~~~~~~~~~

test_tree
    args:
        1. learned_tree: A DefaultDict decision tree as returned by build_decision_tree function.
        2. file_path: A string containing file path for a comma separated file containing data in the same format as 
        data used to train learned_tree.
        3. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        4. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        matches: integer count of examples for which the label in the provided file matches the label assigned by 
        learned_tree
        total: total number of examples read from the file.

~~~~~~~~~~

Ensemble Learning
-----------------
-----------------

AdaBoost
--------


This is an implementation of the AdaBoost algorithm for CS6350 at University of Utah in Spring 2019. Utilized 
decision stumps built on the ID3 framework with Entropy information gain.

Build Hypothesis
~~~~~~~~~~

ada_boost
    args:
        1. examples: data set as a list of examples, each represented by a list of values and a label
        2. iterations: Integer for number of iterations of the algorithm to run.
        4. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        5. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        hypothesis: A list of 3-tuples, of the form (tree_i, alpha_i, error_i), the ith decision stump and its 
        relevant statistics.

~~~~~~~~~~

Get Label
~~~~~~~~~

get_label
    args:
        1. hypothesis: A list of 3-tuples, of the form (tree_i, alpha_i, error_i), the ith decision stump and its 
        information. 
        2. example: A single example in the form of a list of values.
        3. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        4. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        The label assigned this example according to the input hypothesis.

~~~~~~~~~

Test Hypothesis
~~~~~~~~~~

test_ada_boost_hypothesis
    args:
        1. hypothesis: A list of 3-tuples, of the form (tree_i, alpha_i, error_i), the ith decision stump and its 
        information. 
        2. example_params: A string containing file path for a comma separated file containing data in the same format
         as 
        data used to train learned_tree, or a dataset in the format of list of examples input as lists.
        3. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        4. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        results: List of tuples, where each tuple is of the form (matches_i, total_i) for each tree aggregated in 
        the AdaBoost hypothesis.

~~~~~~~~~~

--------

Bagged Trees
-------



This is an implementation of a bagged trees algorithm for CS6350 at University of Utah in Spring 2019. Utilizes 
decision trees built in the ID3 framework on bagged samples from the training data set.

Build Hypothesis
~~~~~~~~~~

bagged_trees
    args:
        1. examples: String containing file path.
        2. iterations: integer for maximum depth of decision tree.
        3. sample_size: integer size of bagged sample for each tree construction.
        4. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        5. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        A list of (tree, % accuracy) tuples

~~~~~~~~~~

Get Label
~~~~~~~~~

get_label
    args:
        1. hypothesis: A list of 3-tuples, of the form (tree_i, alpha_i, error_i), the ith decision stump and its 
        information. 
        2. example: A single example in the form of a list of values.
        3. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        4. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        The label assigned this example according to the input hypothesis.

~~~~~~~~~

Test Hypothesis
~~~~~~~~~~

test_bagged_tree_hypothesis
    args:
        1. hypothesis: A list of 3-tuples, of the form (tree_i, alpha_i, error_i), the ith decision stump and its 
        information. 
        2. example_params: A string containing file path for a comma separated file containing data in the same format
         as 
        data used to train learned_tree, or a dataset in the format of list of examples input as lists.
        3. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        4. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        hypothesis: List of tuples, where each tuple is of the form (matches_i, total_i) for each tree aggregated in 
        the bagged tree hypothesis.

~~~~~~~~~~
-------

Random Forest
-------


This is an implementation of a random forest algorithm for CS6350 at University of Utah in Spring 2019. Utilizes 
decision trees built in the ID3 framework on random attribute sets on bagged samples from the training data set.

Build Hypothesis
~~~~~~~~~~

random_forest
    args:
        1. examples: String containing file path.
        2. iterations: integer for maximum depth of decision tree.
        3. sample_size: integer size of bagged sample for each tree construction.
        4. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        5. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
        6. feature_size: integer number of features to construct trees.
    return:
        A list of (tree, % accuracy) tuples

~~~~~~~~~~

Get Label
~~~~~~~~~

get_label
    args:
        1. hypothesis: A list of 3-tuples, of the form (tree_i, alpha_i, error_i), the ith decision stump and its 
        information. 
        2. example: A single example in the form of a list of values.
        3. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        4. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        The label assigned this example according to the input hypothesis.

~~~~~~~~~

Test Hypothesis
~~~~~~~~~~

test_bagged_tree_hypothesis
    args:
        1. hypothesis: A list of 3-tuples, of the form (tree_i, alpha_i, error_i), the ith decision stump and its 
        information. 
        2. example_params: A string containing file path for a comma separated file containing data in the same format
         as 
        data used to train learned_tree, or a dataset in the format of list of examples input as lists.
        3. numeric_cols: List of integer indices indicating which columns of the input data should be treated as 
        numeric. Any column not listed will be considered categorical and discrete.
        4. missing_identifier: String within examples indicating a missing value. 'NULL' or 'unknown' are common 
        examples.
    return:
        hypothesis: List of tuples, where each tuple is of the form (matches_i, total_i) for each tree aggregated in 
        the bagged tree hypothesis.

~~~~~~~~~~

-------

Linear Classifiers
-----------------
-----------------

Least Mean Squares
------------------

This is an implementation of a least mean squares algorithm for CS6350 at University of Utah in Spring 2019.

Build Hypothesis
~~~~~~~~~~

least_mean_squares
    args:
        1. example_param: String containing file path.
        2. batch_size: number of examples to review before updating weight vector
        2. iterations: Iterations to run the algorithm
        3. sample_size: integer size of bagged sample for each tree construction.
        4. learning_constant: float learning constant for calculating weight vectors.
    return:
        A list of (learning constant, [weight vectors], bool convergence variable) tuples

~~~~~~~~~~

Get Label
~~~~~~~~~

get_label
    args:
        1. weight: A weight vector
        2. example: A single example in the form of a list of values.
    return:
        The label assigned this example according to the input hypothesis.

~~~~~~~~~

Test Hypothesis
~~~~~~~~~~

test_lms
    args:
        1. hypothesis: Weight vector.
        2. example_params: A string containing file path for a comma separated file containing data in the same format
         as 
        data used to train learned_tree, or a dataset in the format of list of examples input as lists.
    return:
        Loss of test on the given example set.

~~~~~~~~~~

Perceptron
----------

Coming Soon


Support Vector Machine
----------

Coming Soon


Artificial Neural Network
----------

Coming Soon
