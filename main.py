import numpy as np
import argparse
import csv
from sklearn.cluster import Birch
import matplotlib.pyplot as plt

import pandas as pd
import plotly.plotly
import plotly.graph_objs as go
import seaborn as sns

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


def compute_clusters(data: List) -> np.ndarray:
	print("--->Computing clusters")
	birch = Birch(
		branching_factor=50,
		n_clusters=5,
		threshold=0.3,
		copy=True,
		compute_labels=True
	)

	birch.fit(data)
	predictions = np.array(birch.predict(data))
	return predictions


def show_results(data: np.ndarray, labels: np.ndarray) -> None:
	labels = np.reshape(labels, (1, labels.size))
	data = np.concatenate((data, labels.T), axis=1)
	
	# Seaborn plot
	facet = sns.lmplot(
		data=pd.DataFrame(data, columns=["Income", "Spending", "Label"]), 
		x="Income", 
		y="Spending", 
		hue='Label', 
		fit_reg=False, 
		legend=True, 
		legend_out=True
	)

	# Pure matplotlib plot
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# scatter = ax.scatter(data[:,0], data[:, 1], c=data[:, 2], s=50)
	# ax.set_title("Clusters")
	# ax.set_xlabel("Income")
	# ax.set_ylabel("Spending")
	# plt.colorbar(scatter)
	plt.show()


def show_data_corelation(data=None, csv_file_name=None):
	data_set = None
	if csv_file_name is None:
		cor = np.corrcoef(data)
		print("Corelation matrix:")
		print(cor)
	else:
		data_set = pd.read_csv(csv_file_name)
		print(data_set.describe())
		data_set = data_set[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
		cor = data_set.corr()
	sns.heatmap(cor, square=True)
	plt.show()
	return data_set


def main(args) -> None:
	data = load_data(args.data_file)
	filtered_data = np.array([[item[3], item[4]] for item in data])

	data_set = None #Alternative data loaded using pandas
	if args.describe == True:
		data_set = show_data_corelation(csv_file_name=args.data_file)

	filtered_data = np.array(filtered_data).astype(np.float64)
	labels = compute_clusters(filtered_data)
	show_results(filtered_data, labels)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Do some clustering")
	parser.add_argument("--data-file", type=str, default="Mall_Customers.csv", help="dataset file name")
	parser.add_argument("--describe", type=bool, default=False, help="describe the dataset")
	args = parser.parse_args()
	main(args)