import csv

def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(int(row['ds'])) # Changed type float->int and ds-dataset size
            y.append(int(row['sec'])) # Changed type float->int and sec-seconds
    return x, y
