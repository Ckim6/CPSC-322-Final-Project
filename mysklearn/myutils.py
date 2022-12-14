"""
Programmer: Caden Kim
Class: CPSC 322, Fall 2022
Programming Assignment #7
11/19/22

Description: This program stores functions
"""
import csv
import math
import numpy as np
import pandas as pd
import mysklearn.myevaluation as myevaluation


def compute_euclidean_distance(v1, v2):
    """computes the distance of v1 and v2 at each instance

    Args:
        v1: dataset of first attribute
        v2: dataset of second attribute

    Returns:
        dist: distances for each instance
    """
    if type(v1[0]) == str:  # is a string
        dist = np.sqrt(sum(1 if v1[i] == v2[i] else 0 for i in range(len(v1))))
    else:
        dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

    return dist


def get_frequencies(col):
    """finds frequencies of categorical groupings of data

    Args:
        table: dataset
        header: column names
        col_name: column name to be extracted

    Returns:
        values: groupings of column
        counts: subdatasets of groupings
    """
    col.sort()

    values = []
    counts = []
    for value in col:
        if value in values:
            counts[-1] += 1
        else:
            values.append(value)
            counts.append(1)

    return values, counts


def load_csv(filename):
    """loads file from location

    Args:
        filename: file name

    Returns:
        header: column names
        table: dataset
    """
    with open(filename, 'r') as file:
        csvread = csv.reader(file)

        header = []
        header = next(csvread)

        table = []
        for row in csvread:
            table.append(row)

    return header, table

def correct_header(filename):
    header, table = load_csv(filename)
    header = table[0]
    return header

def correct_table(filename):
    header, table = load_csv(filename)
    table.pop(0)
    return table

def compute_random_subset(header, f):
    """used for holdout method selection

    Args:
        header: column names
        f: num att for tree

    Returns:
       values_copy[:f]: shuffled list
    """
    # there is a function np.random.choice()
    values_copy = header[:]  # shallow copy
    np.random.shuffle(values_copy)  # in place shuffle
    return values_copy[:f]

def get_column(table, header, col_name):
    """Extracts column from table

    Args:
        table: dataset
        header: column names
        col_name: column name to be extracted

    Returns:
        col: list of column values
    """
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def group_by(table, header, groupby_col_name):
    groupby_col_index = header.index(groupby_col_name)
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col)))

    group_subtables = [[] for _ in group_names]

    for row in table:
        groupby_val = row[groupby_col_index]
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(row.copy())

    return group_names, group_subtables

def find_column(X_train, y_train, col_index, ci):
    """Extracts column from table

    Args:
        table: dataset
        header: column names
        col_name: column name to be extracted

    Returns:
        col: list of column values
    """
    col = []
    for i, _ in enumerate(y_train):
        if y_train[i] == ci:
            value = X_train[i][col_index]
            col.append(value)

    return col


def find1_column(X_train, col_index):
    """Extracts column from table

    Args:
        table: dataset
        header: column names
        col_name: column name to be extracted

    Returns:
        col: list of column values
    """
    col = []
    for row in X_train:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def get_labels(list_of_val):
    """ returns unique values as labels and their freq
        Args:
            list_of_val(list of str): list of values
        Returns:
            label(list of str): list of unique values as labels
            freq(list of int): frequency of the unique val parallel to labels
    """
    label = []
    freq = []
    for val in list_of_val:
        if val in label:
            idx = label.index(val)
            freq[idx] += 1
        else:
            label.append(val)
            freq.append(1)
    return label, freq

def find_max(label, freqs):
    """ find max finds the label associated with the highest frequency given a list of frequencies
        Args:
            label(list of str): list of labels
            freqs(list of int): frequency of each label(parallel to labels)
        Returns:
            max_label(str): label of the highest frequency
    """
    max_f = 0
    max_label = ""
    for i, freq in enumerate(freqs):
        if freq > max_f:
            max_f = freq
            max_label = label[i]
    return max_label

def get_groups_in_col(data, header, column_name):
    group_labels = []
    col_index = header.index(column_name)
    for row in data:
        if row[col_index] not in group_labels:
            group_labels.append(row[col_index])
    return group_labels

def get_average(col1, col2):
    list = []
    for i in range(len(col1)):
        data = col1[i] + col2[i]
        data = data / 2
        list.append(data)
    return list

def get_average_four_col(col1, col2, col3, col4):
    list = []
    for i in range(len(col1)):
        data = col1[i] + col2[i] + col3[i] + col4[i]
        data = data / 4
        list.append(data)
    return list


def discretized_values(X_test):
    """converts numerical data into classifications

    Args:
        X_test: testing data

    Returns:
        discretized_train: classified X_test
    """
    discretized_train = []
    for i, _ in enumerate(X_test):
        if X_test[i] >= 45:
            discretized_train.append(10)
        elif X_test[i] >= 37:
            discretized_train.append(9)
        elif X_test[i] >= 31:
            discretized_train.append(8)
        elif X_test[i] >= 27:
            discretized_train.append(7)
        elif X_test[i] >= 24:
            discretized_train.append(6)
        elif X_test[i] >= 20:
            discretized_train.append(5)
        elif X_test[i] >= 17:
            discretized_train.append(4)
        elif X_test[i] >= 15:
            discretized_train.append(3)
        elif X_test[i] >= 14:
            discretized_train.append(2)
        else:
            discretized_train.append(1)
    return discretized_train


def accuracy(actual, predicted):
    """calculates accuracy of predicted to real values

    Args:
        actual: real classifications
        predicted: predicted classifications

    Returns:
        accurate_list: accuracy of each instance
    """
    count = 0
    for i, _ in enumerate(predicted):
        if predicted[i] == actual[i]:
            count += 1
    accurate_list = count / len(predicted)
    return accurate_list


def normalize(datalist):
    """normalizes data set by (each instance - min) / range

    Args:
        datalist: list of data

    Returns:
       normalized_list: normalized version on dataset
       min_val: smallest value in list
       range_Val: range of values in list
    """
    min_val = datalist[0]
    max_val = datalist[0]
    for i in datalist:
        if i < min_val:
            min_val = i
        if i > max_val:
            max_val = i
    range_val = max_val - min_val

    normalized_list = []
    for i in datalist:
        norm = (i - min_val) / range_val
        normalized_list.append(norm)

    return normalized_list, min_val, range_val


def matrices_survived_total(knn_matrix):
    """puts the totals in the matix

    Args:
        knn_matrix: matrix

    Returns:
       knn_matix: matrix with totals
    """
    yes_val = 0
    no_val = 0
    yes_val1 = 0
    no_val1 = 0
    for i, _ in enumerate(knn_matrix):
        for j in range(len(knn_matrix)):
            if i == 0:
                yes_val += knn_matrix[i][j]
            elif i == 1:
                no_val += knn_matrix[i][j]
            if j == 0:
                yes_val1 += knn_matrix[i][j]
            elif j == 1:
                no_val1 += knn_matrix[i][j]

    knn_matrix[0].append(yes_val)
    knn_matrix[1].append(no_val)
    knn_matrix.append([yes_val1, no_val1, yes_val1+no_val1])
    knn_matrix[0].insert(0, "yes")
    knn_matrix[1].insert(0, "no")
    knn_matrix[2].insert(0, "Total")

    return knn_matrix


def matrices_winner_total(knn_matrix):
    """puts the totals in the matix

    Args:
        knn_matrix: matrix

    Returns:
       knn_matix: matrix with totals
    """
    yes_val = 0
    no_val = 0
    yes_val1 = 0
    no_val1 = 0
    for i, _ in enumerate(knn_matrix):
        for j in range(len(knn_matrix)):
            if i == 0:
                yes_val += knn_matrix[i][j]
            elif i == 1:
                no_val += knn_matrix[i][j]
            if j == 0:
                yes_val1 += knn_matrix[i][j]
            elif j == 1:
                no_val1 += knn_matrix[i][j]

    knn_matrix[0].append(yes_val)
    knn_matrix[1].append(no_val)
    knn_matrix.append([yes_val1, no_val1, yes_val1+no_val1])
    knn_matrix[0].insert(0, "H")
    knn_matrix[1].insert(0, "A")
    knn_matrix[2].insert(0, "Total")

    return knn_matrix

def matrices_playoff_binary_total(knn_matrix):
    """puts the totals in the matix

    Args:
        knn_matrix: matrix

    Returns:
       knn_matix: matrix with totals
    """
    yes_val = 0
    no_val = 0
    yes_val1 = 0
    no_val1 = 0
    for i, _ in enumerate(knn_matrix):
        for j in range(len(knn_matrix)):
            if i == 0:
                yes_val += knn_matrix[i][j]
            elif i == 1:
                no_val += knn_matrix[i][j]
            if j == 0:
                yes_val1 += knn_matrix[i][j]
            elif j == 1:
                no_val1 += knn_matrix[i][j]

    knn_matrix[0].append(yes_val)
    knn_matrix[1].append(no_val)
    knn_matrix.append([yes_val1, no_val1, yes_val1+no_val1])
    knn_matrix[0].insert(0, "MP")
    knn_matrix[1].insert(0, "NP")
    knn_matrix[2].insert(0, "Total")

    return knn_matrix

def matrices_playoff_total(knn_matrix):
    np_val = 0
    wc_val = 0
    conf_val = 0
    div_val = 0
    sbL_val = 0
    sbW_val = 0

    np_val1 = 0
    wc_val1 = 0
    conf_val1 = 0
    div_val1 = 0
    sbL_val1 = 0
    sbW_val1 = 0
    for i, _ in enumerate(knn_matrix):
        for j in range(len(knn_matrix)):
            if i == 0:
                np_val += knn_matrix[i][j]
            elif i == 1:
                wc_val += knn_matrix[i][j]
            elif i == 2:
                conf_val += knn_matrix[i][j]
            elif i == 3:
                div_val += knn_matrix[i][j]
            elif i == 4:
                sbL_val += knn_matrix[i][j]
            elif i == 5:
                sbW_val += knn_matrix[i][j]
            if j == 0:
                np_val1 += knn_matrix[i][j]
            elif j == 1:
                wc_val1 += knn_matrix[i][j]
            elif j == 2:
                conf_val1 += knn_matrix[i][j]
            elif j == 3:
                div_val1 += knn_matrix[i][j]
            elif j == 4:
                sbL_val1 += knn_matrix[i][j]
            elif j == 5:
                sbW_val1 += knn_matrix[i][j]

    knn_matrix[0].append(np_val)
    knn_matrix[1].append(wc_val)
    knn_matrix[2].append(conf_val)
    knn_matrix[3].append(div_val)
    knn_matrix[4].append(sbL_val)
    knn_matrix[5].append(sbW_val)
    knn_matrix.append([np_val1, wc_val1, conf_val1, div_val1, sbL_val1, sbW_val1, np_val1+wc_val1+conf_val1+div_val1+sbL_val1+sbW_val1])
    knn_matrix[0].insert(0, "NP")
    knn_matrix[1].insert(0, "Lost WC")
    knn_matrix[2].insert(0, "Lost Conf")
    knn_matrix[3].insert(0, "Lost Div")
    knn_matrix[4].insert(0, "Lost SB")
    knn_matrix[5].insert(0, "Won SB")
    knn_matrix[6].insert(0, "Total")

    return knn_matrix

def select_attribute(instances, attributes):
    """selects the attribute index to partition on

    Args:
        instances: available instances
        attribute: available attributes to select from

    Returns:
       attributes[rand_index]: index of attribute
    """
    select_min_entropy = []
    for i in attributes:
        attribute_types = []
        # find all attribute instance types
        for row in instances:
            if row[i] not in attribute_types:
                attribute_types.append(row[i])
        attribute_instances = [[] for _ in attribute_types]
        # find amount for each attribute
        for row in instances:
            index_att = attribute_types.index(row[i])
            attribute_instances[index_att].append(1)

        class_types = []
        for values in instances:
            if values[-1] not in class_types:
                class_types.append(values[-1])
        class_type_check = [[[] for _ in class_types] for _ in attribute_types]

        for j, _ in enumerate(instances):
            class_type_check[attribute_types.index(
                instances[j][i])][class_types.index(instances[j][-1])].append(1)

        # calculate smallest E_new
        enew = 0
        for entropy_att, _ in enumerate(class_type_check):
            entropy = 0
            for class_entropy in range(len(class_type_check[entropy_att])):
                val_instance = sum(
                    class_type_check[entropy_att][class_entropy])
                einstance = val_instance / \
                    sum(attribute_instances[entropy_att])
                if einstance != 0:
                    entropy += -1 * einstance * math.log(einstance, 2)
            enew += entropy * \
                sum(attribute_instances[entropy_att]) / len(instances)
        select_min_entropy.append(enew)

    min_index = select_min_entropy.index(min(select_min_entropy))
    return attributes[min_index]

def compute_bootstrapped_sample(table):
    """selects the attribute index to partition on

    Args:
        table: remainder set 

    Returns:
       X_train, y_train, X_test, y_test: split
    """
    n = len(table)
    train_set = []
    for _ in range(n):
        # Return random integers from low (inclusive) to high (exclusive)
        rand_index = np.random.randint(0, n)
        train_set.append(table[rand_index])

    validation_set = []
    for i in range(n):
        if table[i] not in train_set:
            validation_set.append(table[i])

    X_train = [row[0:-1] for row in train_set]
    y_train = [row[-1] for row in train_set]
    X_test = [row[0:-1] for row in validation_set]
    y_test = [[row[-1]] for row in validation_set]

    return X_train, y_train, X_test, y_test

def find_majority(index, table):
    """finds majority instance

    Args:
        index: column index
        table: instance

    Returns:
       majority_vote: majority instance
    """
    unique_instances = []
    for row in table:
        if row[index] not in unique_instances:
            unique_instances.append(row[index])

    count_instances = [[] for _ in unique_instances]

    for row in table:
        count_instances[unique_instances.index(row[index])].append(1)

    sum_instances = []
    for row in count_instances:
        sum_instances.append(sum(row))

    majority_vote = unique_instances[sum_instances.index(max(sum_instances))]

    return majority_vote


def partition_instances(instances, split_attribute, X_train):
    """partitions list in dictionary type

    Args:
        instances: available isntances to be patitioned
        split_attribute: attribute that will be partitioned on

    Returns:
       partitions: instance partitions
    """
    # lets use a dictionary
    partitions = {}  # key (string): value (subtable)
    # att_index = header.index(split_attribute) # e.g. 0 for level
    attribute_domains = {}
    for l, _ in enumerate(X_train[0]):
        no_repeats = []
        for row in X_train:
            if str(row[l]) not in no_repeats:
                no_repeats.append(str(row[l]))
        attribute_domains[l] = no_repeats

    att_index = split_attribute
    # e.g. ["Junior", "Mid", "Senior"]
    att_domain = attribute_domains[att_index]
    for att_value in att_domain:
        partitions[att_value] = []
        # task: finish
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions


def all_same_class(instances):
    """checks if all instances have the same label

    Args:
        instances: instances
        class_index: label value being checked

    Returns:
       True or False
    """
    check_same = instances[0][-1]
    for attribute_vals in instances:
        if attribute_vals[-1] != check_same:
            return False
    return True


def second_case(att_partition, current_instances, value_subtree, tree):
    """does majority vote for leaf

    Args:
        att_partition: instances to be partitioned
        current_instances: available attributes to partition
        value_subtree: subtree
        tree: tree
    """
    classifiers = []
    for value_class in att_partition:
        if value_class[-1] not in classifiers:
            classifiers.append(value_class[-1])
    # find amount for each classifier

    find_majority = [[] for _ in classifiers]
    for value_class in att_partition:
        find_majority[classifiers.index(value_class[-1])].append(1)

    # find max amount
    max_val = 0
    for count in find_majority:
        total_sum = sum(count)
        if total_sum > max_val:
            majority_rule = classifiers[find_majority.index(count)]

    leaf_node = ["Leaf", majority_rule, len(
        att_partition), len(current_instances)]
    value_subtree.append(leaf_node)
    tree.append(value_subtree)


def tdidt(current_instances, available_attributes, X_train):
    """recursively builds decision tree

    Args:
        current_instances: instances to be partitioned
        available_attributes: available attributes to partition

    Returns:
       tree: the updated tree
    """
    # select an attribute to split on
    attribute = select_attribute(current_instances, available_attributes)
    available_attributes.remove(attribute)
    tree = ["Attribute", "att" + str(attribute)]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, attribute, X_train)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():
        value_subtree = ["Value", att_value]

        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            leaf_node = ["Leaf", att_partition[0][-1],
                         len(att_partition), len(current_instances)]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            second_case(att_partition, current_instances, value_subtree, tree)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            return None

        else:  # the previous conditions are all false... recurse!!
            subtree = tdidt(
                att_partition, available_attributes.copy(), X_train)
            if subtree is None:
                second_case(att_partition, current_instances,
                            value_subtree, tree)
            else:
                value_subtree.append(subtree)
                tree.append(value_subtree)
    return tree


def tdidt_predict(header, tree, instance):
    """predicts instances using decisions tree

    Args:
        header: attribute labels
        tree: tree after fit() called
        instance: X_test instance

    Returns:
       tree[1]: classification at leaf
    """
    # recursively traverse tree to make a prediction
    # are we at a leaf node (base case) or attribute node?
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]  # label
    # we are at an attribute
    # find attribute value match for instance
    # for loop
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            return tdidt_predict(header, value_list[2], instance)


def tdidt_rules(header, tree):
    """computes rules for tree

    Args:
        header: attribute labels
        tree: tree after fit() called
    """
    info_type = tree[0]
    if info_type == "Leaf":
        print("THEN class = " + tree[1])
        return

    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        print("if att" + str(att_index) + " == ", value_list[1])
        if i != len(tree):
            print("AND")

        return tdidt_rules(header, value_list[2])