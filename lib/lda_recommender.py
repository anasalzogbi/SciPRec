#!/usr/bin/env python
"""
This module provides functionalities for lda based recommender
"""
from util.data_parser import DataParser
from lib.configuration_manager import ConfigurationManager
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from util.top_similar import TopSimilar as TopRecommendations


class LDARecommender(object):

	DATASET = 'dummy'


	def __init__(self):
		self.config_manager = ConfigurationManager()
		paper_presentation = 'words'
		self.topics_num = self.config_manager.get_number_of_topics()
		self.topics_num = 5
		self.parser = DataParser(self.DATASET, self.topics_num, paper_presentation)
		self.documents = self.parser.get_document_word_distribution()
		#print(self.documents.shape)
		self.ratings = self.parser.get_ratings_matrix()
		self.profiles = []
		self.mu = 0.000001
		self.k_folds = 5
		self.train_indices, self.test_indices = self.get_kfold_indices()
		self._train()
	

	def _train(self):
		for user in range(self.ratings.shape[0]):
			for fold in range(self.k_folds):
				self.fold_train_indices, self.fold_test_indices = self.get_fold_indices(fold, self.train_indices, self.test_indices)
				train_indices = self.fold_train_indices[user]
				test_indices = self.fold_test_indices[user]
				liked_items = train_indices
				corpus = []
				#print(liked_items)
				for liked_item in liked_items:
					if self.ratings[user][liked_item] == 1:
						corpus.append(self.documents[liked_item])
				print("corpus")
				print(np.array(corpus).shape)

				## train
				print("training")
				lda = LatentDirichletAllocation(n_topics=self.topics_num, max_iter=10,
									learning_method='online',
									learning_offset=50., random_state=0,
									verbose=0)

				corpus = np.array(corpus)
				lda.fit_transform(corpus)
				user_profile = lda.components_.T
				print(user_profile.shape)
				print("trained")
				print("building model")
				documents_model = self.build_documents_model(corpus, test_indices)
				print("model built")
				## evaluate
				top_predictions = TopRecommendations(200)
				for index, test_document in enumerate(self.test_indices):
					## Compute SKL
					min_skl = np.inf
					for i in range(user_profile.shape[1]):
						skl = 0
						print("topic")
						print(i)
						for word in range(user_profile.shape[0]):
							print(documents_model[:, index])
							print(np.log(user_profile[:,i] / documents_model[:, index]).sum())
							matrix_1 = np.log(user_profile[:,i] / documents_model[:, index]) * user_profile[word, i] * 0.5
							matrix_2 = np.log(documents_model[:, index]/ user_profile[:, i]) * documents_model[word][index] * 0.5
							skl += np.sum(matrix_1 + matrix_2)
						if min_skl > skl:
							min_skl = skl

					similarity = 1 / min_skl
					top_predictions.insert(test_document, similarity)
				k = 10
				dcg = 0
				idcg = 0
				mrr = 0
				recommendation_indices = top_predictions.get_indices()
				for pos_index, index in enumerate(recommendation_indices):
					hit_found = False
					dcg += (self.test_data[user][index] / np.log2(pos_index + 2))
					idcg += 1 / np.log2(pos_index + 2)
					if self.ratings[user][index] == 1 and mrr == 0.0:
						mrr = 1.0 / (pos_index + 1) * 1.0
					if pos_index + 1 == k:
						break
					if idcg != 0:
						break
				recall_at_200 = self.calculate_recall(top_predictions, user, 20)

	def calculate_recall(self, top_predictions, user, k):
		recommendation_indices = top_predictions.get_indices()
		hit_count = 0
		for pos_index, index in enumerate(recommendation_indices):
			hit_count += self.ratings[user][index]
		return hit_count / k

	def build_documents_model(self, corpus, test_documents):
		researchers_test_word_count = np.sum(self.documents[test_documents])
		researchers_word_count = np.sum(corpus, axis=0)
		researchers_train_word_count = np.sum(researchers_word_count)
		denom = (researchers_test_word_count + self.mu)

		## loop on all words
		document_word_presentation = np.zeros((self.documents.shape[1], len(test_documents)))

		for i in range(self.documents.shape[1]):
			## loop on all test documents
			word_train_occurences = researchers_word_count[i]
			numerator = (self.mu * word_train_occurences) / researchers_train_word_count
			for index, document in enumerate(test_documents):
				## nocc(w, dj)
				nocc = self.documents[document][i]
				document_word_presentation[i, index] = (nocc + (numerator)) / (denom)
		return document_word_presentation


	def get_fold(self, fold_num, fold_train_indices, fold_test_indices):
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
		index = fold_num - 1
		for ctr in range(self.ratings.shape[0]):
			current_train_fold_indices.append(fold_train_indices[index])
			current_test_fold_indices.append(fold_test_indices[index])
			index += self.k_folds
		return self.generate_kfold_matrix(current_train_fold_indices, current_test_fold_indices)

	def get_fold_indices(self, fold_num, fold_train_indices, fold_test_indices):
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
		index = fold_num - 1
		for ctr in range(self.ratings.shape[0]):
			current_train_fold_indices.append(fold_train_indices[index])
			current_test_fold_indices.append(fold_test_indices[index])
			index += self.k_folds
		return (current_train_fold_indices, current_test_fold_indices)

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

		self.fold_test_indices = test_indices
		self.fold_train_indices = train_indices

		return train_indices, test_indices

	def generate_kfold_matrix(self, train_indices, test_indices):
		"""
		Returns a training set and a training set matrix for one fold.
		This method is to be used in conjunction with get_kfold_indices()

		:param int[] train_indices array of train set indices.
		:param int[] test_indices array of test set indices.
		:returns: Training set matrix and Test set matrix.
		:rtype: 2-tuple of 2d numpy arrays
		"""
		train_matrix = np.zeros(self.ratings.shape)
		test_matrix = np.zeros(self.ratings.shape)
		for user in range(train_matrix.shape[0]):
			train_matrix[user, train_indices[user]] = self.ratings[user, train_indices[user]]
			test_matrix[user, test_indices[user]] = self.ratings[user, test_indices[user]]
		return train_matrix, test_matrix