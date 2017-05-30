
import numpy as np




def get_kfold_indices(ratings, k_folds):
		"""
		returns the indices for rating matrix for each kfold split. Where each test set
		contains ~1/k of the total items a user has in their digital library.

		:returns: a list of all indices of the training set and test set.
		:rtype: list of lists
		"""

		np.random.seed(42)

		train_indices = []
		test_indices = []

		for user in range(ratings.shape[0]):

			# Indices for all items in the rating matrix.
			item_indices = np.arange(ratings.shape[1])

			# Indices of all items in user's digital library.
			rated_items_indices = ratings[user].nonzero()[0]
			mask = np.ones(len(ratings[user]), dtype=bool)
			mask[[rated_items_indices]] = False
			# Indices of all items not in user's digital library.
			non_rated_indices = item_indices[mask]

			# Shuffle all rated items indices
			np.random.shuffle(rated_items_indices)

			# Size of 1/k of the total user's ratings
			size_of_test = int(round((1.0 / k_folds) * len(rated_items_indices)))

			# 2d List that stores all the indices of each test set for each fold.
			test_ratings = [[] for x in range(k_folds)]

			counter = 0
			np.random.shuffle(non_rated_indices)
			# List that stores the number of indices to be added to each test set.
			num_to_add = []

			# create k different folds for each user.
			for index in range(k_folds):
				if index == k_folds - 1:

					test_ratings[index] = np.array(rated_items_indices[counter:len(rated_items_indices)])
				else:
					test_ratings[index] = np.array(rated_items_indices[counter:counter + size_of_test])
				counter += size_of_test

				# adding unique zero ratings to each test set
				num_to_add.append(int((ratings.shape[1] / k_folds) - len(test_ratings[index])))
				if index > 0 and num_to_add[index] != num_to_add[index - 1]:
					addition = non_rated_indices[index * (num_to_add[index - 1]):
															(num_to_add[index - 1] * index) + num_to_add[index]]
				else:
					addition = non_rated_indices[index * (num_to_add[index]):num_to_add[index] * (index + 1)]

				test_ratings[index] = np.append(test_ratings[index], addition)
				test_indices.append(test_ratings[index])

				# for each user calculate the training set for each fold.
				train_index = rated_items_indices[~np.in1d(rated_items_indices, test_ratings[index])]
				mask = np.ones(len(ratings[user]), dtype=bool)
				mask[[np.append(test_ratings[index], train_index)]] = False

				train_ratings = np.append(train_index, item_indices[mask])
				train_indices.append(train_ratings)

		return train_indices, test_indices

def get_user_fold_indices(user, fold_num, train_indices_shared, test_indices_shared, k_folds):
	return (train_indices_shared[user * k_folds + (fold_num-1)], test_indices_shared[user * k_folds + (fold_num)])


def generate_user_kfold_matrix(user, user_train_indices, user_test_indices, ratings_shared):
	"""
	Returns a training set and a training set matrix for one fold.
	This method is to be used in conjunction with get_kfold_indices()

	:param int[] train_indices array of train set indices.
	:param int[] test_indices array of test set indices.
	:returns: Training set matrix and Test set matrix.
	:rtype: 2-tuple of 2d numpy arrays
	"""
	train_matrix = np.zeros(ratings_shared.shape[1])
	test_matrix = np.zeros(ratings_shared.shape[1])
	train_matrix[user_train_indices] = ratings_shared[user, user_train_indices]
	test_matrix[user_test_indices] = ratings_shared[user, user_test_indices]
	return train_matrix, test_matrix


def get_user_fold_indices_and_data(user, fold_num, train_indices_shared, test_indices_shared, ratings_shared, k_folds):
	"""
	Returns train and test data for a given fold number

	:param int fold_num the fold index to be returned
	:param int[] fold_train_indices: A list of the indicies of the training fold.
	:param int[] fold_test_indices: A list of the indicies of the testing fold.
	:returns: tuple of training and test data
	:rtype: 2-tuple of 2d numpy arrays
	"""
	user_train_fold_indices, user_test_fold_indices = get_user_fold_indices(user, fold_num, train_indices_shared, test_indices_shared, k_folds)
	user_train_matrix, user_test_matrix = generate_user_kfold_matrix(user, user_train_fold_indices, user_test_fold_indices, ratings_shared)
	return user_train_fold_indices, user_test_fold_indices, user_train_matrix, user_test_matrix




def get_fold_indices(fold_num, train_indices_shared, test_indices_shared, ratings_shared, k_folds):
	"""
	Returns train and test data for a given fold number

	:param int fold_num the fold index to be returned
	:param int[] fold_train_indices: A list of the indicies of the training fold.
	:param int[] fold_test_indices: A list of the indicies of the testing fold.
	:returns: tuple of training and test data
	:rtype: 2-tuple of 2d numpy arrays
	"""

	current_train_fold_indices = []
	current_test_fold_indices = []
	index = fold_num
	for ctr in range(ratings_shared.shape[0]):
		current_train_fold_indices.append(train_indices_shared[index])
		current_test_fold_indices.append(test_indices_shared[index])
		index += k_folds
	return (current_train_fold_indices, current_test_fold_indices)

def get_fold_indices_and_data(fold_num, train_indices_shared, test_indices_shared, ratings_shared, k_folds):
	"""
	Returns train and test data for a given fold number

	:param int fold_num the fold index to be returned
	:param int[] fold_train_indices: A list of the indicies of the training fold.
	:param int[] fold_test_indices: A list of the indicies of the testing fold.
	:returns: tuple of training and test data
	:rtype: 2-tuple of 2d numpy arrays
	"""
	train_fold_indices, test_fold_indices = get_fold_indices(fold_num, train_indices_shared, test_indices_shared, ratings_shared, k_folds)
	train_matrix, test_matrix = generate_kfold_matrix(train_fold_indices, test_fold_indices, ratings_shared)
	return train_fold_indices, test_fold_indices, train_matrix, test_matrix

def generate_kfold_matrix(train_indices, test_indices, ratings_shared):
	"""
	Returns a training set and a training set matrix for one fold.
	This method is to be used in conjunction with get_kfold_indices()

	:param int[] train_indices array of train set indices.
	:param int[] test_indices array of test set indices.
	:returns: Training set matrix and Test set matrix.
	:rtype: 2-tuple of 2d numpy arrays
	"""
	train_matrix = np.zeros(ratings_shared.shape)
	test_matrix = np.zeros(ratings_shared.shape)
	for user in range(train_matrix.shape[0]):
		train_matrix[user, train_indices[user]] = ratings_shared[user, train_indices[user]]
		test_matrix[user, test_indices[user]] = ratings_shared[user, test_indices[user]]
	return train_matrix, test_matrix


def get_test_documents(test_indices, user, document_matrix_shared):
	documents = []
	indices = []
	for index in test_indices[user]:
		documents.append(document_matrix_shared[index])
		indices.append(index)
	return np.array(documents), np.array(indices)
