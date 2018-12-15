import numpy as np
import argparse
import csv
from sklearn.cluster import Birch

from typing import  Tuple, Dict, List

def load_data(file_name) -> List[List]:
    print("--->Loading csv file")

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        data = []

        for line in csv_reader:
            if line_count == 0:
                print(f'Column names: [{", ".join(line)}]')
            else:
                data.append(line)
            line_count += 1 

    print(f'Loaded {line_count} records')
    return data

def main(args) -> None:
    data = load_data(args.data_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do some clustering")
    parser.add_argument("--data-file", type=str, default="Mall_Customers.csv", help="dataset file name")
    args = parser.parse_args()
    main(args)