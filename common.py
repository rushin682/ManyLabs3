import csv


def import_data(file_name):
    data = []
    with open(file_name, mode='rt') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data


def write_data(file_name, all_rows):
    with open(file_name, mode="w", encoding="utf8", newline="") as file:
        writer = csv.writer(file)
        for i in all_rows:
            writer.writerow(i)
    file.close()