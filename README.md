This is a machine learning library developed by John Jacobson for CS5350/6350 in University of Utah

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

Coming Soon

--------

Bagged Trees
-------

Coming Soon

-------

Random Forest
-------

Coming Soon

-------

Linear Classifiers
-----------------
-----------------

Least Mean Squares
------------------

Coming Soon

Perceptron
----------

Coming Soon
