import csv
import numpy as np
import string
import re


def age(ages):
    all_rows = list()
    pattern = r"\d+\.?\d*"
    all_rows.append([ages[0]])
    for a in range(1, len(ages)):
        nums_str = re.findall(pattern, ages[a])
        if len(nums_str) == 0:
            all_rows.append(['NA'])
        else:
            all_rows.append([nums_str[len(nums_str) - 1]])
    write_data("age.csv", all_rows)
    return all_rows


def anagram(words):
    """
    :param words: A row * 4 array
    :return: A row * 4 array. For column 0 and 1, if correct append 'TRUE', if 'NA' append 'NA', otherwise append 'FALSE';
            For column 2 and 3, if 'NA' append 'NA', otherwise append 'FALSE'
    """
    check_list = ['party', 'fatal']
    all_rows = list()
    all_rows.append(words[0])
    for r in range(1, len(words)):
        res = []
        for i in range(4):
            if i == 0 or i == 1:
                if words[r, i] == check_list[i]:
                    res.append("TRUE")
                elif words[r, i] == '"NA':
                    res.append('NA')
                else:
                    res.append("FALSE")
            else:
                if words[r, i] != 'NA':
                    res.append("FALSE")
                else:
                    res.append('NA')
        all_rows.append(res)
    write_data("anagram.csv", all_rows)
    return all_rows


def attention(atte):
    """
    :param atte: A row * 2 array
    :return: A row * 1 array. For atte, if column 0 is not NA, the result is FALSE; If column 0 is NA and column 1 is "I read
            the instructions", the result is TRUE; Otherwise, FALSE
    """
    all_rows = list()
    all_rows.append(["attention"])
    for r in range(1, len(atte)):
        if atte[r, 1] == "I read the instructions" and atte[r, 0] == 'NA':
            all_rows.append(["TRUE"])
        elif atte[r, 0] != 'NA' or (atte[r, 1] != "I read the instructions" and atte[r, 1] != 'NA'):
            all_rows.append(["FALSE"])
        else:
            all_rows.append(['NA'])
    write_data("attention.csv", all_rows)
    return all_rows


def backcount(nums):
    """
    :param nums: A row * 10 array
    :return: A row * 10 array. If the number is correct, append 'TRUE'; If incorrect, append 'FALSE'; Otherwise append 'NA'
    """
    check_list = np.array(['357', '330', '354', '351', '348', '345', '342', '339', '336', '333'])
    all_rows = list()
    all_rows.append(nums[0])
    for r in range(1, len(nums)):
        res = []
        for i in range(len(nums[r])):
            if nums[r, i] == check_list[i]:
                res.append("TRUE")
            elif nums[r, i] == 'NA':
                res.append('NA')
            else:
                res.append("FALSE")
        all_rows.append(res)
    write_data("backcount.csv", all_rows)
    return all_rows


def grade(arr, best_or_worst):
    """
    :param arr: A row * 5 array
    :param best_or_worst: A row * 6 array, bestgrade1 separate into 2 columns,
    :return:
    """
    all_rows = list()
    all_rows.append(['season', 'year', best_or_worst + 'grade2', best_or_worst + 'grade3', best_or_worst + 'grade4', best_or_worst + 'grade5'])
    for r in range(1, len(arr)):
        res = []
        for i in range(len(arr[r])):
            if arr[r, i] != 'NA':
                if i == 0:
                    time_str = arr[r, i].lower()
                    if "spring" in time_str:
                        res.append('1')
                    elif "summer" in time_str:
                        res.append('2')
                    elif "fall" in time_str:
                        res.append('3')
                    elif "winter" in time_str:
                        res.append('4')
                    else:
                        res.append('5')
                    if "2013" in time_str or "2014" in time_str:
                        res.append('1')
                    else:
                        res.append('2')
                elif i == 1:
                    pattern = r"\d+\.?\d*"
                    nums_str = re.findall(pattern, arr[r, i])
                    if "A" in arr[r, i].upper() or (len(nums_str) > 0 and (float(nums_str[0]) >= 90.0 or float(nums_str[0]) == 4.0)):
                        res.append('1')
                    elif "B" in arr[r, i].upper() or (len(nums_str) > 0 and (80.0 <= float(nums_str[0]) < 90.0 or 3.0 <= float(nums_str[0]) < 4.0)):
                        res.append('2')
                    elif "C" in arr[r, i].upper() or (len(nums_str) > 0 and (70.0 <= float(nums_str[0]) < 80.0 or 2.0 <= float(nums_str[0]) < 3.0)):
                        res.append('3')
                    elif "D" in arr[r, i].upper() or (len(nums_str) > 0 and (60.0 <= float(nums_str[0]) < 70.0 or 1.0 <= float(nums_str[0]) < 2.0)):
                        res.append('4')
                    elif "F" in arr[r, i].upper() or (len(nums_str) > 0 and (float(nums_str[0]) < 60 or float(nums_str[0]) < 1.0)):
                        res.append('5')
                    else:
                        res.append('6')
                elif i == 2 or i == 4:
                    res.append((int(arr[r, i]) - 5.5) / 4.5)
                else:
                    res.append((int(arr[r, i]) - 4) / 3)
            else:
                res.append('NA')
                if i == 0:
                    res.append('NA')
        all_rows.append(res)
    write_data(best_or_worst + "grade.csv", all_rows)
    return all_rows



def div3filler(nums):
    """
    :param nums: A row * 1 array with string
    :return: A row * 1 array, if the occurrence counts match, the column is TRUE, otherwise FALSE
    """
    total_3 = 3
    total_6 = 2
    total_9 = 3
    all_rows = list()
    all_rows.append(nums[0])
    for r in range(1, len(nums)):
        count_3 = 0
        count_6 = 0
        count_9 = 0
        num_string = nums[r, 0].replace(' ', '')
        if num_string != 'NA':
            for i in range(len(num_string)):
                if num_string[i] not in string.punctuation:
                    if num_string[i] == '3':
                        count_3 += 1
                    elif num_string[i] == '6':
                        count_6 += 1
                    elif num_string[i] == '9':
                        count_9 += 1
                    else:
                        break
            if count_3 == total_3 and count_6 == total_6 and count_9 == total_9:
                all_rows.append(['TRUE'])
            else:
                all_rows.append(['FALSE'])
        else:
            all_rows.append(['NA'])
    write_data("div3filler.csv", all_rows)
    return all_rows


def ethnicity(arr):
    all_rows = list()
    all_rows.append(arr[0])
    for r in range(1, len(arr)):
        if arr[r, 0] != 'NA' and is_number(arr[r, 0]):
            all_rows.append([(int(arr[r, 0]) - 4.5) / 3.5])
        elif arr[r, 0] == 'NA':
            all_rows.append(['NA'])
        else:
            all_rows.append([(8 - 4.5) / 3.5])
    write_data("ethnicity.csv", all_rows)
    return all_rows


def gender(arr):
    all_rows = list()
    all_rows.append(arr[0])
    for r in range(1, len(arr)):
        if arr[r, 0] != 'NA':
            if arr[r, 0] == '1':
                all_rows.append([1])
            else:
                all_rows.append([-1])
        else:
            all_rows.append(['NA'])
    write_data("gender.csv", all_rows)
    return all_rows


def position_ratio(arr, letter):
    all_rows = list()
    all_rows.append(arr[0])
    for r in range(1, len(arr)):
        res = []
        if arr[r, 0] == '1':
            res.append(1)
        elif arr[r, 0] == '2':
            res.append(-1)
        else:
            res.append('NA')

        try:
            pattern = r"\d+\.?\d*"
            if arr[r, 1] != 'NA' and (len(re.findall(pattern, arr[r, 1])) and 1 <= int(arr[r, 1]) < 100):
                num = int(arr[r, 1])
                if num == 10:
                    res.append(0)
                elif num > 10:
                    res.append(num / 10 - 1)
                else:
                    res.append(1 - 10 / num)
            else:
                res.append('NA')
        except:
            res.append('NA')
        all_rows.append(res)
    write_data(letter + "position_ratio.csv", all_rows)
    return all_rows


def major(majors):
    print("pending")


def mcdv(arr):
    all_rows = list()
    all_rows.append(arr[0])
    for r in range(1, len(arr)):
        res = []
        for i in range(len(arr[r])):
            if arr[r, i] != 'NA':
                res.append(int(arr[r, i]) / 3)
            else:
                res.append('NA')
        all_rows.append(res)
    write_data("mcdv.csv", all_rows)
    return all_rows


def mcfiller(arr):
    all_rows = list()
    all_rows.append(arr[0])
    for r in range(1, len(arr)):
        res = []
        for i in range(len(arr[r])):
            if i == 0 or i == 2:
                if arr[r, i] != 'NA':
                    res.append((int(arr[r, i]) - 2.5) / 1.5)
                else:
                    res.append('NA')
            else:
                if arr[r, i] != 'NA':
                    res.append((int(arr[r, i]) - 2) / 2)
                else:
                    res.append('NA')
        all_rows.append(res)
    write_data("mcfiller.csv", all_rows)
    return all_rows


def mcmost_mcsome(arr):
    all_rows = list()
    all_rows.append(arr[0])
    for r in range(1, len(arr)):
        res = []
        for i in range(len(arr[r])):
            if arr[r, i] != 'NA':
                if arr[r, i] == '1':
                    res.append(1)
                else:
                    res.append(-1)
            else:
                res.append('NA')
        all_rows.append(res)
    write_data("mcmost_mcsome.csv", all_rows)
    return all_rows


def numeric_value(arr, file_name, normal):
    all_rows = list()
    all_rows.append(arr[0])
    for r in range(1, len(arr)):
        res = []
        for i in range(len(arr[r])):
            if arr[r, i] != 'NA':
                res.append((int(arr[r, i]) - normal) / (normal - 1))
            else:
                res.append('NA')
        all_rows.append(res)
    write_data(file_name, all_rows)
    return all_rows


def pate(arr):
    all_rows = list()
    all_rows.append(arr[0])
    for r in range(1, len(arr)):
        res = []
        for i in range(len(arr[r])):
            if arr[r, i] != 'NA':
                if i == 0 or i == 1:
                    res.append((int(arr[r, i]) - 3) / 2)
                elif i == 2:
                    res.append(int(arr[r, i]) - 2)
                else:
                    res.append((int(arr[r, i]) - 3.5) / 2.5)
            else:
                res.append('NA')
        all_rows.append(res)
    write_data("pate.csv", all_rows)
    return all_rows


def tempest(arr):
    all_rows = list()
    all_rows.append(arr[0])
    pattern = r"\d+\.?\d*"
    temp = []
    for r in range(1, len(arr)):
        nums_str = re.findall(pattern, arr[r, 0])
        if len(nums_str) >= 1:
            avg = 0
            for s in nums_str:
                num = float(s)
                if num <= 40:
                    num = num * 1.8 + 32
                avg += num
            temp.append(avg / len(nums_str))
            arr[r, 0] = avg / len(nums_str)
        else:
            arr[r, 0] = 'NA'
    temp_arr = np.array(temp)
    temp_avg = np.average(temp_arr)  # avg is 71.9
    for r in range(1, len(arr)):
        res = []
        for i in range(len(arr[r])):
            if arr[r, i] != 'NA' or is_number(arr[r, i]):
                if i == 0:
                    number = float(arr[r, i])
                    print(number, temp_avg)
                    if number >= temp_avg:
                        res.append(number / temp_avg - 1)
                    else:
                        res.append(1 - temp_avg / number)
                else:
                    res.append((float(arr[r, i]) - 4) / 3)
            else:
                res.append('NA')
        all_rows.append(res)
    # write_data("temp.csv", all_rows)
    return all_rows


def write_data(file_name, all_rows):
    with open(file_name, mode="w", encoding="utf8", newline="") as file:
        writer = csv.writer(file)
        for i in all_rows:
            writer.writerow(i)
    file.close()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


data = []
with open("ML3AllSites.csv", mode="rt") as file:
    reader = csv.reader(file)
    for row in reader:
        # print(row)
        data.append(row)
file.close()
data_arr = np.array(data)


age(data_arr[:, 4])
# anagram(data_arr[:, 5:9])
# attention(data_arr[:, 9:11])
# backcount(data_arr[:, 11:21])
# grade(data_arr[:, 21:26], "best")
# numeric_value(data_arr[:, 26:36], "big5.csv", 4)
# div3filler(data_arr[:, 36:37])
# numeric_value(data_arr[:, 37:42], "elm.csv", 5)
# ethnicity(data_arr[:, 42:43])
# gender(data_arr[:, 44:45])
# numeric_value(data_arr[:, 50:65], "intrinsic.csv", 2.5)
# position_ratio(data_arr[:, 65:67], "k")
# position_ratio(data_arr[:, 68:70], "l")
# mcdv(data_arr[:, 71:73])
# mcfiller(data_arr[:, 73:76])
# mcmost_mcsome(data_arr[:, 76:86])
# numeric_value(data_arr[:, 86:88], "mood.csv", 4)
# numeric_value(data_arr[:, 88:94], "nfc.csv", 3)
# position_ratio(data_arr[:, 94:96], "n")
# pate(data_arr[:, 96:101])
# position_ratio(data_arr[:, 101:103], "r")
# numeric_value(data_arr[:, 103:104], "sarcasm.csv", 3.5)
# numeric_value(data_arr[:, 104:105], "selfesteem.csv", 4)
# numeric_value(data_arr[:, 105:109], "stress.csv", 3)
# tempest(data_arr[:, 109:115])
# position_ratio(data_arr[:, 115:117], "v")
# grade(data_arr[:, 117:122], "worst")

