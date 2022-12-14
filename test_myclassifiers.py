"""
Programmer: Caden Kim
Class: CPSC 322, Fall 2022
Programming Assignment #7
11/19/22
Description: This program tests functions
"""

import numpy as np
from scipy import stats
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyNaiveBayesClassifier, \
    MyDecisionTreeClassifier, \
    MyRandomForestClassifier
from sklearn.linear_model import LinearRegression 

# TODO: copy your test_myclassifiers.py solution from PA4-6 here

def high_low_discretizer(value):
    if value <= 100:
        return "low"
    return "high"

def test_kneighbors_classifier_kneighbors():
    function_kneighbor = MyKNeighborsClassifier()

    # from in-class #1  (4 instances)
    X_test = [0.33, 1]
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    # desk check
    test_distances = [0.67, 1.00, 1.05]
    test_neighbor_indices = [0, 2, 3]
    # function check
    function_kneighbor.fit(X_train_class_example1, y_train_class_example1)
    func_dist, funct_indices = function_kneighbor.kneighbors(X_test)

    assert np.allclose(func_dist, test_distances)
    assert np.allclose(funct_indices, test_neighbor_indices)

    # from in-class #2 (8 instances)
    # assume normalized
    X_test = [2, 3]
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    # desk check
    test_distances = [1.41, 1.41, 2.00]
    test_neighbor_indices = [0, 4, 6]
    # function check
    function_kneighbor.fit(X_train_class_example2, y_train_class_example2)
    func_dist, funct_indices = function_kneighbor.kneighbors(X_test)

    assert np.allclose(func_dist, test_distances)
    assert np.allclose(funct_indices, test_neighbor_indices)

    # from Bramer
    X_test = [9.1, 11.0]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    # desk check
    test_distances = [0.61, 1.24, 2.20]
    test_neighbor_indices = [6, 5, 7]
    # function check
    function_kneighbor.fit(X_train_bramer_example, y_train_bramer_example)
    func_dist, funct_indices = function_kneighbor.kneighbors(X_test)

    assert np.allclose(func_dist, test_distances)
    assert np.allclose(funct_indices, test_neighbor_indices)


def test_kneighbors_classifier_predict():
    function_kpredict = MyKNeighborsClassifier()

    # from in-class #1  (4 instances)
    X_test = [0.33, 1]
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    # desk check
    test_y_predicted = ["good"]
    # function check
    function_kpredict.fit(X_train_class_example1, y_train_class_example1)
    y_predicted = function_kpredict.predict(X_test)

    assert y_predicted == test_y_predicted

    # from in-class #2 (8 instances)
    # assume normalized
    X_test = [2, 3]
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    # desk check
    test_y_predicted = ["yes"]
    # function check
    function_kpredict.fit(X_train_class_example2, y_train_class_example2)
    y_predicted = function_kpredict.predict(X_test)

    assert y_predicted == test_y_predicted

    # from Bramer
    X_test = [9.1, 11.0]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    # desk check
    test_y_predicted = ["+"]
    # function check
    function_kpredict.fit(X_train_bramer_example, y_train_bramer_example)
    y_predicted = function_kpredict.predict(X_test)

    assert y_predicted == test_y_predicted


def test_dummy_classifier_fit():
    dummy_class_fit = MyDummyClassifier()
    X_train = list(range(100))

    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    test_most_common_label = "yes"
    dummy_class_fit.fit(X_train, y_train)
    assert test_most_common_label == dummy_class_fit.most_common_label

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    test_most_common_label = "no"
    dummy_class_fit.fit(X_train, y_train)
    assert test_most_common_label == dummy_class_fit.most_common_label

    y_train = list(np.random.choice(["go", "slow", "stop"], 100, replace=True, p=[0.3, 0.6, 0.1]))
    test_most_common_label = "slow"
    dummy_class_fit.fit(X_train, y_train)
    assert test_most_common_label == dummy_class_fit.most_common_label


def test_dummy_classifier_predict():
    dummy_class_predict = MyDummyClassifier()
    X_train = list(range(100))
    X_test = [1,2,3,4]

    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_class_predict.fit(X_train, y_train)
    test_dummy_predict = [["yes"], ["yes"], ["yes"], ["yes"]]
    for i in range(len(test_dummy_predict)):
        assert dummy_class_predict.predict(X_test)[i] == test_dummy_predict[i]

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_class_predict.fit(X_train, y_train)
    test_dummy_predict = [["no"], ["no"], ["no"], ["no"]]
    for i in range(len(test_dummy_predict)):
        assert dummy_class_predict.predict(X_test)[i] == test_dummy_predict[i]

    y_train = list(np.random.choice(["go", "slow", "stop"], 100, replace=True, p=[0.3, 0.6, 0.1]))
    dummy_class_predict.fit(X_train, y_train)
    test_dummy_predict = [["slow"], ["slow"], ["slow"], ["slow"]]
    for i in range(len(test_dummy_predict)):
        assert dummy_class_predict.predict(X_test)[i] == test_dummy_predict[i]


# in-class Naive Bayes example (lab task #1)
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# RQ5 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no",
            "yes", "yes", "yes", "yes", "yes", "no", "yes"]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain", "class"]
train_table = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]

y_train_table = ["on time", "on time", "on time", "late", "on time", "very late",
    "on time", "on time", "very late", "on time", "cancelled", "on time",
    "late", "on time", "very late", "on time", "on time", "on time", "on time",
    "on time"]

def test_naive_bayes_classifier_fit():
    expected_priors = {'no': 0.38, 'yes': 0.62}
    expected_posteriors = {'no': {0: {1: 0.67, 2: 0.33}, 1: {5: 0.67, 6: 0.33}},
                          'yes': {0: {1: 0.8, 2: 0.2}, 1: {5: 0.4, 6: 0.6}}}

    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(X_train_inclass_example, y_train_inclass_example)
    assert expected_priors == naive_bayes_fit.priors
    assert expected_posteriors == naive_bayes_fit.posteriors


    expected_priors = {'no': 0.33, 'yes': 0.67}
    expected_posteriors = {'no': {0: {1: 0.6, 2: 0.4}, 1: {1: 0.2, 2: 0.4, 3: 0.4}, 2: {'excellent': 0.6, 'fair': 0.4}},
                        'yes': {0: {1: 0.2, 2: 0.8}, 1: {1: 0.3, 2: 0.4, 3: 0.3}, 2: {'excellent': 0.3, 'fair': 0.7}}}

    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(X_train_iphone, y_train_iphone)
    assert expected_priors == naive_bayes_fit.priors
    assert expected_posteriors == naive_bayes_fit.posteriors


    expected_priors = {'cancelled': 0.05, 'late': 0.1, 'on time': 0.7, 'very late': 0.15}

    expected_posteriors = {'cancelled': {0: {'saturday': 1.0, 'weekday': 0.0, 'holiday': 0.0, 'sunday': 0.0},
                                        1: {'spring': 1.0, 'winter': 0.0, 'summer': 0.0, 'autumn': 0.0},
                                        2: {'high': 1.0, 'none': 0.0, 'normal': 0.0}, 3: {'heavy': 1.0, 'none': 0.0, 'slight': 0.0}},
                          'late': {0: {'saturday': 0.5, 'weekday': 0.5, 'holiday': 0.0, 'sunday': 0.0},
                                  1: {'spring': 1.0, 'winter': 0.0, 'summer': 0.0, 'autumn': 0.0},
                                  2: {'high': 0.5, 'none': 0.5, 'normal': 0.0}, 3: {'heavy': 0.5, 'none': 0.5, 'slight': 0.0}},
                          'on time': {0: {'saturday': 0.14, 'weekday': 0.14, 'holiday': 0.07, 'sunday': 0.64},
                                      1: {'spring': 0.14, 'winter': 0.29, 'summer': 0.43, 'autumn': 0.14},
                                      2: {'high': 0.29, 'none': 0.36, 'normal': 0.36},
                                      3: {'heavy': 0.07, 'none': 0.36, 'slight': 0.57}},
                          'very late': {0: {'saturday': 1.0, 'weekday': 0.0, 'holiday': 0.0, 'sunday': 0.0},
                                        1: {'spring': 0.33, 'winter': 0.67, 'summer': 0.0, 'autumn': 0.0},
                                        2: {'high': 0.33, 'none': 0.67, 'normal': 0.0},
                                        3: {'heavy': 0.67, 'none': 0.33, 'slight': 0.0}}}

    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(train_table, y_train_table)
    assert expected_priors == naive_bayes_fit.priors
    assert expected_posteriors == naive_bayes_fit.posteriors

def test_naive_bayes_classifier_predict():
    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(X_train_inclass_example, y_train_inclass_example)
    X_test = [[1,5]]
    expected_predictions = ["yes"]
    assert naive_bayes_fit.predict(X_test) == expected_predictions

    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(X_train_iphone, y_train_iphone)
    X_test = [[2,2,"fair"], [1,1,"excellent"]]
    expected_predictions = ["yes", "no"]
    assert naive_bayes_fit.predict(X_test) == expected_predictions

    X_test = [["weekday", "winter", "high", "heavy"], ["weekday", "winter", "high", "heavy"], ["sunday", "summer", "normal", "slight"]]
    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(train_table, y_train_table)
    expected_predictions = ["on time", "on time", "on time"]
    assert naive_bayes_fit.predict(X_test) == expected_predictions

def test_decision_tree_classifier_fit():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    # interview dataset
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True",
                         "False", "True", "True", "True", "True", "True", "False"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview)

    # attribute values are now sorted by index
    tree_interview = \
        ['Attribute', 'att0',
         ['Value', 'Senior',
          ['Attribute', 'att2',
           ['Value', 'no',
            ['Leaf', 'False', 3, 5]],
           ['Value', 'yes',
            ['Leaf', 'True', 2, 5]]]],
         ['Value', 'Mid',
          ['Leaf', 'True', 4, 14]],
         ['Value', 'Junior',
          ['Attribute', 'att3',
           ['Value', 'no',
            ['Leaf', 'True', 3, 5]],
           ['Value', 'yes',
            ['Leaf', 'False', 2, 5]]]]]
    assert tree_interview == decision_tree.tree

    X_train_iphone = [
        ["1", "3", "fair"],
        ["1", "3", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "fair"],
        ["2", "1", "fair"],
        ["2", "1", "excellent"],
        ["2", "1", "excellent"],
        ["1", "2", "fair"],
        ["1", "1", "fair"],
        ["2", "2", "fair"],
        ["1", "2", "excellent"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no",
                      "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_iphone, y_train_iphone)

    tree_iphone = \
        ['Attribute', 'att0',
            ['Value', '1',
                ['Attribute', 'att1',
                    ['Value', '3',
                        ['Leaf', 'no', 2, 5]],
                    ['Value', '2',
                        ['Attribute', 'att2',
                            ['Value', 'fair',
                                ['Leaf', 'no', 1, 2]],
                            ['Value', 'excellent',
                                ['Leaf', 'yes', 1, 2]]]],
                    ['Value', '1',
                        ['Leaf', 'yes', 1, 5]]]],
            ['Value', '2',
                ['Attribute', 'att2',
                    ['Value', 'fair',
                        ['Leaf', 'yes', 6, 10]],
                    ['Value', 'excellent',
                        ['Leaf', 'no', 4, 10]]]]]
    assert tree_iphone == decision_tree.tree

def test_decision_tree_classifier_predict():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    # interview dataset
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True",
                         "False", "True", "True", "True", "True", "True", "False"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview)
    X_test = [["Junior", "Java", "yes", "no"],
              ["Junior", "Java", "yes", "yes"]]
    expected_predict = ["True", "False"]
    predicted = decision_tree.predict(X_test)
    assert expected_predict == predicted

    X_train_iphone = [
        ["1", "3", "fair"],
        ["1", "3", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "fair"],
        ["2", "1", "fair"],
        ["2", "1", "excellent"],
        ["2", "1", "excellent"],
        ["1", "2", "fair"],
        ["1", "1", "fair"],
        ["2", "2", "fair"],
        ["1", "2", "excellent"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no",
                      "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_iphone, y_train_iphone)
    X_test = [["2", "2", "fair"], ["1", "1", "excellent"]]
    expected_predict = ["yes", "yes"]
    predicted = decision_tree.predict(X_test)
    assert expected_predict == predicted

# interview dataset
X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True",
                     "False", "True", "True", "True", "True", "True", "False"]

def test_random_forest_fit():
    np.random.seed(0)
    random_forest = MyRandomForestClassifier(3, 2, 3)
    random_forest.fit(X_train_interview, y_train_interview)

    expected_m_tree = [['Attribute','att2', ['Value','yes', ['Attribute', 'att0', ['Value', 'Senior', ['Leaf', 'True', 5, 9]], ['Value', 'Mid', ['Leaf', 'True', 2, 9]], ['Value', 'Junior', ['Leaf', 'False', 2, 9]]]], ['Value', 'no', ['Leaf', 'False', 5, 14]]], ['Attribute', 'att1', ['Value', 'Python', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 7, 8]], ['Value', 'yes', ['Leaf', 'False', 1, 8]]]], ['Value', 'R',
     ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 2, 3]], ['Value', 'yes', ['Leaf', 'False', 1, 3]]]], ['Value', 'Java', ['Leaf', 'False', 3, 14]]]]

    assert expected_m_tree == random_forest.m_forest_vis


def test_random_forest_predict():
    np.random.seed(0)
    random_forest = MyRandomForestClassifier(3, 2, 3)
    random_forest.fit(X_train_interview, y_train_interview)

    X_test = [["Junior", "Java", "yes", "no"],
              ["Junior", "Java", "yes", "yes"]]

    expected_predict = ['F', 'F']
    y_predicted = random_forest.predict(X_test)
    assert expected_predict == y_predicted