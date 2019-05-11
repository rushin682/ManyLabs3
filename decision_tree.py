import numpy as np
import math
import operator
import common
import copy


def convert_whole_data_grade(arr):
    data = []
    for i in range(1, len(arr)):
        tmp = list()
        for j in range(2, 6):
            if arr[i][j] == 'NA':
                tmp.append('NA')
            else:
                if j == 2:
                    tmp.append(arr[i][j])
                else:
                    if float(arr[i][j]) <= -0.5:
                        tmp.append('-1')
                    elif -0.5 < float(arr[i][j]) < 0.5:
                        tmp.append('0')
                    else:
                        tmp.append('1')
        if arr[i][0] == 'NA':
            tmp.append('NA')
        else:
            tmp.append(arr[i][0] + arr[i][1])
        data.append(tmp)
    return np.array(data)


def recover_data_of_grade(constructed_data, title, original_data):
    recover_data = list()
    recover_data.append(title)
    for r in range(len(constructed_data)):
        tmp = list()
        # process season and year
        if constructed_data[r][4][0] == '1':
            tmp.append('Spring')
        elif constructed_data[r][4][0] == '2':
            tmp.append('Summer')
        elif constructed_data[r][4][0] == '3':
            tmp.append('Autumn')
        elif constructed_data[r][4][0] == '4':
            tmp.append('Winter')
        else:
            tmp.append('High school')
        if constructed_data[r][4][1] == '1':
            tmp.append(np.random.choice(['2013', '2014']))
        else:
            tmp.append(np.random.choice(['2012', '2011', '2010', '2009', '2008', 'current']))
        # process bestgrade2
        tmp.append(chr(ord(constructed_data[r][0]) + 16))
        # process bestgrade3
        if original_data[r][1] == 'NA':
            if constructed_data[r][1] == '-1':
                tmp.append(np.random.choice(['1', '2', '3']))
            elif constructed_data[r][1] == '0':
                tmp.append(np.random.choice(['4', '5', '6', '7']))
            else:
                tmp.append(np.random.choice(['8', '9', '10']))
        else:
            tmp.append(round(float(original_data[r][1]) * 4.5 + 5.5))
        # process bestgrade4
        if original_data[r][2] == 'NA':
            if constructed_data[r][2] == '-1':
                tmp.append(np.random.choice(['1', '2']))
            elif constructed_data[r][2] == '0':
                tmp.append(np.random.choice(['3', '4', '5']))
            else:
                tmp.append(np.random.choice(['6', '7']))
        else:
            tmp.append(round(float(original_data[r][2]) * 3 + 4))
        # process bestgrade5
        if original_data[r][3] == 'NA':
            if original_data[r][3] == '-1':
                tmp.append(np.random.choice(['1', '2', '3']))
            elif original_data[r][3] == '0':
                tmp.append(np.random.choice(['4', '5', '6', '7']))
            else:
                tmp.append(np.random.choice(['8', '9', '10']))
        else:
            tmp.append(round(float(original_data[r][3]) * 4.5 + 5.5))
        recover_data.append(tmp)
    return recover_data


def recover_data_of_gender(constructed_data, title):
    recover_data = list()
    recover_data.append(title)
    for row in constructed_data:
        tmp = list()
        for c in row:
            if c == '-1':
                tmp.append(2)
            else:
                tmp.append(1)
        recover_data.append(tmp)
    return recover_data


def recover_data_of_temp(constructed_data, title, origin_data):
    recover_data = list()
    recover_data.append(title)
    for r in range(len(constructed_data)):
        tmp = list()
        if float(constructed_data[r][5]) <= 0:
            res = round(71.9 / (1 - float(constructed_data[r][5])))
            if res > 100:
                tmp.append(res / 2)
            else:
                tmp.append(res)
        else:
            res = round((float(constructed_data[r][5]) + 1) * 71.9)
            if res > 100:
                tmp.append(res / 2)
            else:
                tmp.append(res)
        for c in range(len(constructed_data[r]) - 1):
            if origin_data[r][c] == 'NA':
                if constructed_data[r][c] == '-1':
                    tmp.append(np.random.choice(['1', '2']))
                elif constructed_data[r][c] == '0':
                    tmp.append(np.random.choice(['3', '4', '5']))
                else:
                    tmp.append(np.random.choice(['6', '7']))
            else:
                tmp.append(round(float(constructed_data[r][c]) * 3 + 4))
        recover_data.append(tmp)
    return recover_data


def construct_data_of_grade(arr, training_ratio=0.7):
    # define the season + year as Y
    data = []
    whole_data = []
    for i in range(1, len(arr)):
        tmp = list()
        for j in range(2, 6):
            if arr[i][j] == 'NA':
                tmp.append('NA')
            else:
                if j == 2:
                    tmp.append(arr[i][j])
                else:
                    if float(arr[i][j]) <= -0.5:
                        tmp.append('-1')
                    elif -0.5 < float(arr[i][j]) < 0.5:
                        tmp.append('0')
                    else:
                        tmp.append('1')
        if arr[i][0] != 'NA':
            tmp.append(arr[i][0] + arr[i][1])
            data.append(tmp)
        else:
            tmp.append('NA')
        whole_data.append(tmp)
    train = [data[i] for i in range(int(len(data) * training_ratio))]
    test = [data[i] for i in range(int(len(data) * training_ratio), len(data))]
    weight = [1 for i in range(len(train))]
    label = arr[0][2:]
    return np.array(train), np.array(test), weight, label, whole_data


def construct_data_of_gender(gender_data, mc_data, start, end, training_ratio=0.7):
    data = []
    whole_data = []
    for i in range(1, len(gender_data)):
        if gender_data[i][0] != 'NA':
            data.append(mc_data[i][start:end] + gender_data[i])
        whole_data.append(mc_data[i][start:end] + gender_data[i])
    train = [data[i] for i in range(int(len(data) * training_ratio))]
    test = [data[i] for i in range(int(len(data) * training_ratio), len(data))]
    weight = [1 for i in range(len(train))]
    label = mc_data[0][start:end] + gender_data[0]
    return np.array(train), np.array(test), weight, label, np.array(whole_data)


def construct_data_of_temp(temp_data, training_ratio=0.7):
    data = []
    whole_data = []
    for i in range(1, len(temp_data)):
        if temp_data[i][0] != 'NA':
            tmp = list()
            for j in range(1, len(temp_data[i])):
                if temp_data[i][j] == 'NA':
                    tmp.append('NA')
                else:
                    if float(temp_data[i][j]) <= -0.5:
                        tmp.append('-1')
                    elif -0.5 < float(temp_data[i][j]) < 0.5:
                        tmp.append('0')
                    else:
                        tmp.append('1')
            if float(temp_data[i][0]) <= 0:
                tmp.append('-1')
            else:
                tmp.append('1')
            data.append(tmp)
        whole_data.append(temp_data[i][1:] + temp_data[i][0:1])
    train = [data[i] for i in range(int(len(data) * training_ratio))]
    test = [data[i] for i in range(int(len(data) * training_ratio), len(data))]
    weight = [1 for i in range(len(train))]
    label = temp_data[0][1:] + temp_data[0][0:1]
    return np.array(train), np.array(test), weight, label, np.array(whole_data)


def calculate_gain(data, weight):
    number_of_features = len(data[0]) - 1
    best_info_gain = -999
    best_feature = -1
    best_chance = []
    for i in range(number_of_features):
        no_missing_data, no_missing_weight = find_no_missing(data, weight)
        rho = len(no_missing_data) / len(data)
        base_ent = calculate_ent(no_missing_data, no_missing_weight)
        feature_value_list = [row[i] for row in no_missing_data]
        unique_values = set(feature_value_list)
        count = [1 for u in range(len(unique_values))]
        flag = 0
        new_ent = 0
        for value in unique_values:
            sub_data, sub_weight = split_data(no_missing_data, i, no_missing_weight, value)
            count[flag] = sum(sub_weight) / sum(no_missing_weight)
            new_ent += count[flag] * calculate_ent(sub_data, sub_weight)
            flag += 1
        info_gain = rho * (base_ent - new_ent)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
            best_chance = count
    return best_feature, best_chance


def calculate_ent(data, weight):
    rows = len(data)
    total_weight = sum(weight)
    value_count = {}
    for r in range(rows):
        current_value = data[r, -1]
        if current_value not in value_count.keys():
            value_count[current_value] = 0
        value_count[current_value] += weight[r]
    ent = 0
    for k in value_count:
        prob = value_count[k] / total_weight
        ent -= prob * math.log(prob)
    return ent


def find_no_missing(data, weight):
    no_missing_data = []
    no_missing_weight = weight[:]
    for r in range(len(data)):
        if 'NA' in data[r]:
            no_missing_weight[r] = 0
        else:
            no_missing_data.append(data[r])
    length = len(no_missing_weight)
    flag = 0
    while flag < length:
        if no_missing_weight[flag] == 0:
            del(no_missing_weight[flag])
            length -= 1
            flag -= 1
        flag += 1
    return np.array(no_missing_data), no_missing_weight


def split_data(features, split_index, weight, ele, chance=[], index=0):
    rows = len(features)
    cols = len(features[0])
    data = []
    new_weight = []
    flag = 0
    for r in range(rows):
        if features[r, split_index] == ele:
            tmp = []
            for c in range(cols):
                if c != split_index:
                    tmp.append(features[r, c])
            data.append(tmp)
            new_weight.append(weight[flag])
        if features[r, split_index] == 'NA':
            tmp = []
            for c in range(cols):
                if c != split_index:
                    tmp.append(features[r, c])
            data.append(tmp)
            new_weight.append(weight[flag] * chance[index])
        flag += 1
    return np.array(data), new_weight


def build_tree(data, weight, label):
    class_list = [row[-1] for row in data]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data[0]) == 1:
        return get_majority(class_list)
    # find the index with maximum gain
    best_feature, best_chance = calculate_gain(data, weight)
    # find the best label
    best_label = label[best_feature]
    tree = {best_label: {}}
    del(label[best_feature])
    # get the values of the feature with the best infomation gain
    best_feature_values = [row[best_feature] for row in data]
    unique_values = set(best_feature_values)
    if 'NA' in unique_values:
        unique_values.remove('NA')
    val_arr = [v for v in unique_values]
    for value in unique_values:
        sub_labels = label[:]
        sub_data, sub_weight = split_data(data, best_feature, weight, value, best_chance, val_arr.index(value))
        tree[best_label][value] = build_tree(sub_data, sub_weight, sub_labels)
    return tree


def get_majority(class_list):
    class_set = {}
    for i in class_list:
        if i not in class_set.keys():
            class_set[i] = 0
        class_set[i] += 1
    sorted_set = sorted(class_set.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_set[0][0]


def calculate_accuracy(tree, test_data, label):
    test_feature = np.delete(test_data, -1, axis=1)
    test_label = [row[-1] for row in test_data]
    correct_count = 0
    for i in range(len(test_data)):
        result = classify(tree, test_feature[i], label)
        if result == test_label[i]:
            correct_count += 1
    accuracy = correct_count / len(test_feature)
    return accuracy


def classify(tree, test_vector, label):
    node = list(tree.keys())[0]
    leaves = tree[node]
    index = label.index(node)
    class_label = ''
    for k in leaves.keys():
        if test_vector[index] == k:
            if type(leaves[k]).__name__ == 'dict':
                class_label = classify(leaves[k], test_vector, label)
            else:
                class_label = leaves[k]
    return class_label


def imputation(tree, missing_row, label, values, res):
    node = list(tree.keys())[0]
    leaves = tree[node]
    index = label.index(node)
    if missing_row[index] == 'NA':
        total = 0
        value = list()
        probability = list()
        for k, v in values[node].items():
            total += v
            value.append(k)
            probability.append(v)
        p = np.array(probability) / total
        v_choice = np.random.choice(value, p=p.ravel())
        missing_row[index] = str(v_choice)
    for key in leaves.keys():
        if missing_row[index] == key:
            if type(leaves[key]).__name__ == 'dict':
                imputation(leaves[key], missing_row, label, values, res)
            else:
                if missing_row[len(label) - 1] == 'NA':
                    missing_row[len(label) - 1] = leaves[key]
    if missing_row[len(label) - 1] == 'NA':
        total = 0
        value = list()
        probability = list()
        for k, v in values[res].items():
            total += v
            value.append(k)
            probability.append(v)
        p = np.array(probability) / total
        v_choice = np.random.choice(value, p=p.ravel())
        missing_row[len(label) - 1] = str(v_choice)


def get_value_set(data, label):
    values = {}
    for c in range(len(data[0])):
        tmp = {}
        for r in range(len(data)):
            if data[r][c] != 'NA':
                if data[r][c] not in tmp.keys():
                    tmp[data[r][c]] = 1
                else:
                    tmp[data[r][c]] += 1
        values[label[c]] = tmp
    return values


if __name__ == '__main__':

    # process bestgrade
    best_grade = common.import_data("bestgrade.csv")
    train_best_data, test_best_data, best_weights, best_labels, bestgrade_all_data = construct_data_of_grade(best_grade, 0.8)
    label_for_best_test = best_labels[:]
    label_for_best_imputation = best_labels[:]
    label_for_best_imputation.append('res')
    bestgrade_value_set = get_value_set(train_best_data, label_for_best_imputation)
    # construct the tree
    best_tree = build_tree(train_best_data, best_weights, best_labels)
    # test the tree
    accuracy = calculate_accuracy(best_tree, test_best_data, label_for_best_test)
    print("Accuracy of bestgrade tree:", accuracy)
    # fill the missing data
    bestgrade_all_data_copy = copy.deepcopy(bestgrade_all_data)
    for data_row in bestgrade_all_data:
        if 'NA' in data_row:
            imputation(best_tree, data_row, label_for_best_imputation, bestgrade_value_set, 'res')
    bestgrade_final_data = recover_data_of_grade(bestgrade_all_data, best_grade[0], bestgrade_all_data_copy)
    # common.write_data("imputation_file/bestgrade_filled.csv", bestgrade_final_data)

    # process worstgrade
    worst_grade = common.import_data("worstgrade.csv")
    train_worst_data, test_worst_data, worst_weights, worst_labels, worstgrade_all_data = construct_data_of_grade(worst_grade, 0.8)
    label_for_worst_test = worst_labels[:]
    label_for_worst_imputation = worst_labels[:]
    label_for_worst_imputation.append('res')
    worstgrade_value_set = get_value_set(train_worst_data, label_for_worst_imputation)
    # construct the tree
    worst_tree = build_tree(train_worst_data, worst_weights, worst_labels)
    # test the tree
    accuracy = calculate_accuracy(worst_tree, test_worst_data, label_for_worst_test)
    print("Accuracy of worstgrade tree:", accuracy)
    # fill the missing data
    worstgrade_all_data_copy = copy.deepcopy(worstgrade_all_data)
    for data_row in worstgrade_all_data:
        if 'NA' in data_row:
            imputation(worst_tree, data_row, label_for_worst_imputation, worstgrade_value_set, 'res')
    worstgrade_final_data = recover_data_of_grade(worstgrade_all_data, worst_grade[0], worstgrade_all_data_copy)
    # common.write_data("imputation_file/worstgrade_filled.csv", worstgrade_final_data)

    # process gender, mcmost, mcsome
    gender = common.import_data("gender.csv")
    mcmost_mcsome = common.import_data("mcmost_mcsome.csv")
    train_gender_mcmost_data, test_gender_mcmost_data, gender_mcmost_weight, gender_mcmost_label, mcmost_whole_data = construct_data_of_gender(gender, mcmost_mcsome, 0, 5, 0.8)
    train_gender_mcsome_data, test_gender_mcsome_data, gender_mcsome_weight, gender_mcsome_label, mcsome_whole_data = construct_data_of_gender(gender, mcmost_mcsome, 5, 10, 0.8)
    label_for_gender_mcmost_test = gender_mcmost_label[:]
    label_for_gender_mcmost_imputation = gender_mcmost_label[:]
    label_for_gender_mcsome_test = gender_mcsome_label[:]
    label_for_gender_mcsome_imputation = gender_mcsome_label[:]
    gender_mcmost_value_set = get_value_set(train_gender_mcmost_data, label_for_gender_mcmost_imputation)
    gender_mcsome_value_set = get_value_set(train_gender_mcsome_data, label_for_gender_mcsome_imputation)
    # construct the tree
    gender_mcmost_tree = build_tree(train_gender_mcmost_data, gender_mcmost_weight, gender_mcmost_label)
    gender_mcsome_tree = build_tree(train_gender_mcsome_data, gender_mcsome_weight, gender_mcsome_label)
    # test the tree
    accuracy = calculate_accuracy(gender_mcmost_tree, test_gender_mcmost_data, label_for_gender_mcmost_test)
    print("Accuracy of gender-mcmost tree:", accuracy)
    accuracy = calculate_accuracy(gender_mcsome_tree, test_gender_mcsome_data, label_for_gender_mcsome_test)
    print("Accuracy of gender-mcsome tree:", accuracy)
    # fill the missing data
    for data_row in mcmost_whole_data:
        if 'NA' in data_row:
            imputation(gender_mcmost_tree, data_row, label_for_gender_mcmost_imputation, gender_mcmost_value_set, 'gender')
    mcmost_final_data = recover_data_of_gender(mcmost_whole_data, mcmost_mcsome[0][0:5] + gender[0])
    for data_row in mcsome_whole_data:
        if 'NA' in data_row:
            imputation(gender_mcsome_tree, data_row, label_for_gender_mcsome_imputation, gender_mcsome_value_set, 'gender')
    mcsome_final_data = recover_data_of_gender(mcsome_whole_data, mcmost_mcsome[0][5:10] + gender[0])
    # common.write_data("imputation_file/mcmost_filled.csv", mcmost_final_data)
    # common.write_data("imputation_file/mcsome_filled.csv", mcsome_final_data)

    # process temp
    temp = common.import_data("temp.csv")
    train_temp_data, test_temp_data, temp_weight, temp_label, temp_whole_data = construct_data_of_temp(temp, 0.8)
    label_for_temp_test = temp_label[:]
    label_for_temp_imputation = temp_label[:]
    temp_value_set = get_value_set(train_temp_data, label_for_temp_imputation)
    # construct the tree
    temp_tree = build_tree(train_temp_data, temp_weight, temp_label)
    # test the tree
    accuracy = calculate_accuracy(temp_tree, test_temp_data, label_for_temp_test)
    print("Accuracy of temp tree:", accuracy)
    # fill the missing data
    temp_whole_data_copy = copy.deepcopy(temp_whole_data)
    for data_row in temp_whole_data:
        if 'NA' in data_row:
            imputation(temp_tree, data_row, label_for_temp_imputation, temp_value_set, 'tempest1')
            if 'NA' in data_row:
                for col in range(len(data_row)):
                    if data_row[col] == 'NA':
                        total = 0
                        value = list()
                        probability = list()
                        for k, v in temp_value_set[label_for_temp_imputation[col]].items():
                            total += v
                            value.append(k)
                            probability.append(v)
                        p = np.array(probability) / total
                        v_choice = np.random.choice(value, p=p.ravel())
                        data_row[col] = str(v_choice)
    temp_final_data = recover_data_of_temp(temp_whole_data, temp[0], temp_whole_data_copy)
    # common.write_data("imputation_file/temp_filled.csv", temp_final_data)