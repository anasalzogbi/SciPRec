import numpy as np
import matplotlib.pyplot as plt
import os

def users_papers_histogram(dataset, path):
	#path = "users.dat"
	u = []
	p = []
	i = 0
	with open (path,'r') as f:
		for line in f:
			u.append(i)
			p.append(len(line.split(" ")))
			i += 1
	#a = np.hstack((u,p))
	#plt.hist(a, bins=400)
	hist, bin_edges = np.histogram(p,bins=range(max(p)), density=True)
	print hist.sum()
	print np.sum(hist * np.diff(bin_edges))
	plt.hist(p,bins=range(max(p)))
	plt.title("Histogram for number of papers in users libraries in {}".format(dataset))
	plt.show()


if __name__ == "__main__":
	dataset_folder = "datasets/Extended_ctr/citeulike_a_extended"
	dataset = "citeulike_a"
	path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), dataset_folder, 'users.dat')

	users_papers_histogram(dataset, path)

