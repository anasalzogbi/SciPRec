from util.data_parser import DataParser
from util.data_dumper import DataDumper
import numpy as np

class DataSplitter(object):

	DATASET = 'dummy'

	def __init__(self):

		self.parser = DataParser(self.DATASET, 0, "keywords")
		self.ratings = self.parser.get_ratings_matrix()
		self.k_folds = 5
		self.train_indices, self.test_indices = self.get_kfold_indices()
		data_dumper = DataDumper(self.DATASET, 'splits')
		data_dumper.save_matrix(self.train_indices, 'train_indices')
		data_dumper.save_matrix(self.test_indices, 'test_indices')

	def get_kfold_indices(self):
		"""
		returns the indices for rating matrix for each kfold split. Where each test set
		contains ~1/k of the total items a user has in their digital library.

		:returns: a list of all indices of the training set and test set.
		:rtype: list of lists
		"""

		np.random.seed(42)

		train_indices = []
		test_indices = []

		for user in range(self.ratings.shape[0]):

			# Indices for all items in the rating matrix.
			item_indices = np.arange(self.ratings.shape[1])

			# Indices of all items in user's digital library.
			rated_items_indices = self.ratings[user].nonzero()[0]
			mask = np.ones(len(self.ratings[user]), dtype=bool)
			mask[[rated_items_indices]] = False
			# Indices of all items not in user's digital library.
			non_rated_indices = item_indices[mask]

			# Shuffle all rated items indices
			np.random.shuffle(rated_items_indices)

			# Size of 1/k of the total user's ratings
			size_of_test = int(round((1.0 / self.k_folds) * len(rated_items_indices)))

			# 2d List that stores all the indices of each test set for each fold.
			test_ratings = [[] for x in range(self.k_folds)]

			counter = 0
			np.random.shuffle(non_rated_indices)
			# List that stores the number of indices to be added to each test set.
			num_to_add = []

			# create k different folds for each user.
			for index in range(self.k_folds):
				if index == self.k_folds - 1:

					test_ratings[index] = np.array(rated_items_indices[counter:len(rated_items_indices)])
				else:
					test_ratings[index] = np.array(rated_items_indices[counter:counter + size_of_test])
				counter += size_of_test

				# adding unique zero ratings to each test set
				num_to_add.append(int((self.ratings.shape[1] / self.k_folds) - len(test_ratings[index])))
				if index > 0 and num_to_add[index] != num_to_add[index - 1]:
					addition = non_rated_indices[index * (num_to_add[index - 1]):
															(num_to_add[index - 1] * index) + num_to_add[index]]
				else:
					addition = non_rated_indices[index * (num_to_add[index]):num_to_add[index] * (index + 1)]

				test_ratings[index] = np.append(test_ratings[index], addition)
				test_indices.append(test_ratings[index])

				# for each user calculate the training set for each fold.
				train_index = rated_items_indices[~np.in1d(rated_items_indices, test_ratings[index])]
				mask = np.ones(len(self.ratings[user]), dtype=bool)
				mask[[np.append(test_ratings[index], train_index)]] = False

				train_ratings = np.append(train_index, item_indices[mask])
				train_indices.append(train_ratings)

		return train_indices, test_indices

DataSplitter()
