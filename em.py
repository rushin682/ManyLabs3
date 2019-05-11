import numpy as np
import common
from matplotlib import pyplot as plt


def plot_data(column, start, end):
    appearance_set = {}
    for r in range(1, len(column)):
        if column[r][0] != 'NA':
            if column[r][0] in appearance_set.keys():
                appearance_set[column[r][0]] += 1
            else:
                appearance_set[column[r][0]] = 1
    x = []
    y = []
    for i in sorted(appearance_set):
        x.append(i)
        y.append(appearance_set[i])
    plt.figure("Figure of ages appearance")
    plt.scatter(x, y)
    # plt.xlim(start, end)
    # plt.ylim((0, 100))
    plt.xticks(np.arange(start, end, 10))
    plt.show()


def construct_grade_data(data):
    data_list = list()
    for i in range(1, len(data)):
        tmp = []
        if data[i][1] != 'NA':
            tmp.append(float(data[i][1]))
        else:
            tmp.append(float('inf'))
        data_list.append(tmp)
    return data_list


def construct_data(data):
    data_list = list()
    for i in range(1, len(data)):
        tmp = []
        for j in range(len(data[0])):
            if data[i][j] != 'NA':
                tmp.append(float(data[i][j]))
            else:
                tmp.append(float('inf'))
        data_list.append(tmp)
    return data_list


def find_null(data):
    null_xy = np.argwhere(np.isnan(data))
    return null_xy


def em(data, loops=50):
    null_xy = find_null(data)
    total = 0
    for x, y in null_xy:
        column = data[:, y]
        column_selected = column[~np.isnan(column)]
        mu = column_selected.mean()  # mean of each column without 'NA'
        sd = column_selected.std()  # std of each column without 'NA'
        column[x] = np.random.normal(loc=mu, scale=sd)
        previous, i = 1, 1
        print("EM for one missing spot:")
        for i in range(loops):
            column_selected = column[~np.isnan(column)]
            mu = column_selected.mean()
            sd = column_selected.std()
            column[x] = np.random.normal(loc=mu, scale=sd)
            delta = (abs(column[x] - previous)) / previous
            # break the loop if the likelihood changes less than 0.1 and loop more than 5 times
            print("loop:", i, ", delta:", delta)
            if i > 5 and delta < 0.1:
                data[x][y] = column[x]
                total += i
                break
            data[x][y] = column[x]
            previous = column[x]
    print("average loop:", total / len(null_xy))
    return data


def recover_data_position_ratio(constructed_data):
    for r in constructed_data:
        if float(r[1]) < 0:
            r[1] = round(10 / (1 - float(r[1])))
        elif float(r[1]) > 0:
            r[1] = round((float(r[1]) + 1) * 10)
        else:
            r[1] = '10'
        if r[0] == 'NA':
            if float(r[1]) <= 10:
                r[0] = '1'
            else:
                r[0] = '2'
        else:
            if r[0] == '-1':
                r[0] = '2'
    return constructed_data


if __name__ == '__main__':
    # process k
    k_data = common.import_data("kposition_ratio.csv")
    k_ratio_list = construct_grade_data(k_data)
    k_ratio_arr = np.array(k_ratio_list)
    for i in range(len(k_ratio_arr)):
        for j in range(len(k_ratio_arr[0])):
            if k_ratio_arr[i][j] == float('inf'):
                # set 'NA' to np.nan
                k_ratio_arr[i][j] = np.nan
    k_ratio = em(k_ratio_arr)
    for i in range(len(k_ratio)):
        if k_data[i][1] == 'NA':
            k_data[i][1] = str(k_ratio[i][0])
    k_recovered_data = recover_data_position_ratio(k_data[1:])
    # common.write_data("imputation_file/kposition_ratio_filled.csv", k_recovered_data)

    # process l
    l_data = common.import_data("lposition_ratio.csv")
    l_ratio_list = construct_grade_data(l_data)
    l_ratio_arr = np.array(l_ratio_list)
    for i in range(len(l_ratio_arr)):
        for j in range(len(l_ratio_arr[0])):
            if l_ratio_arr[i][j] == float('inf'):
                # set 'NA' to np.nan
                l_ratio_arr[i][j] = np.nan
    l_ratio = em(l_ratio_arr)
    for i in range(len(l_ratio)):
        if l_data[i][1] == 'NA':
            l_data[i][1] = str(l_ratio[i][0])
    l_recovered_data = recover_data_position_ratio(l_data[1:])
    # common.write_data("imputation_file/lposition_ratio_filled.csv", l_recovered_data)

    # process r
    r_data = common.import_data("rposition_ratio.csv")
    r_ratio_list = construct_grade_data(r_data)
    r_ratio_arr = np.array(r_ratio_list)
    for i in range(len(r_ratio_arr)):
        for j in range(len(r_ratio_arr[0])):
            if r_ratio_arr[i][j] == float('inf'):
                # set 'NA' to np.nan
                r_ratio_arr[i][j] = np.nan
    r_ratio = em(r_ratio_arr)
    for i in range(len(r_ratio)):
        if r_data[i][1] == 'NA':
            r_data[i][1] = str(r_ratio[i][0])
    r_recovered_data = recover_data_position_ratio(r_data[1:])
    # common.write_data("imputation_file/rposition_ratio_filled.csv", r_recovered_data)

    # process n
    n_data = common.import_data("nposition_ratio.csv")
    n_ratio_list = construct_grade_data(n_data)
    n_ratio_arr = np.array(n_ratio_list)
    for i in range(len(n_ratio_arr)):
        for j in range(len(n_ratio_arr[0])):
            if n_ratio_arr[i][j] == float('inf'):
                # set 'NA' to np.nan
                n_ratio_arr[i][j] = np.nan
    n_ratio = em(n_ratio_arr)
    for i in range(len(n_ratio)):
        if n_data[i][1] == 'NA':
            n_data[i][1] = str(n_ratio[i][0])
    n_recovered_data = recover_data_position_ratio(n_data[1:])
    # common.write_data("imputation_file/nposition_ratio_filled.csv", n_recovered_data)

    # process v
    v_data = common.import_data("vposition_ratio.csv")
    v_ratio_list = construct_grade_data(v_data)
    v_ratio_arr = np.array(v_ratio_list)
    for i in range(len(v_ratio_arr)):
        for j in range(len(v_ratio_arr[0])):
            if v_ratio_arr[i][j] == float('inf'):
                # set 'NA' to np.nan
                v_ratio_arr[i][j] = np.nan
    v_ratio = em(v_ratio_arr)
    for i in range(len(v_ratio)):
        if v_data[i][1] == 'NA':
            v_data[i][1] = str(v_ratio[i][0])
    v_recovered_data = recover_data_position_ratio(v_data[1:])
    # common.write_data("imputation_file/vposition_ratio_filled.csv", v_recovered_data)
    # process age
    age_data = common.import_data("age.csv")
    plot_data(age_data, 0, 60)
    age_data_list = construct_data(age_data)
    age_data_arr = np.array(age_data_list)
    # ages bounded between 15-50
    for i in range(len(age_data_arr)):
        for j in range(len(age_data_arr[0])):
            if age_data_arr[i][j] == float('inf') or float(age_data_arr[i][j]) < 15 or float(age_data_arr[i][j]) > 50:
                age_data_arr[i][j] = np.nan
    age = em(age_data_arr)
    # reconstruct data
    for a in range(len(age)):
        if age[a][0] < 13:
            age[a][0] = 15
        elif age[a][0] > 60:
            age[a][0] = 50
        else:
            age[a][0] = round(age[a][0])
    # common.write_data("imputation_file/age.csv", age)