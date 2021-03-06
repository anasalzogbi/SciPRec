#!/usr/bin/env python
"""
This module provides functionalities for training an svr model
"""
from util.data_parser import DataParser
from util.keyword_extractor import KeywordExtractor
from lib.peer_extractor import PeerExtractor
import numpy as np
from sklearn.svm import LinearSVC as SKSVR
from util.top_similar import TopSimilar as TopRecommendations
from scipy import sparse
import datetime
import sys
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from lib.configuration_manager import ConfigurationManager

class SVR(object):

	DATASET = 'dummy'

	def __init__(self):
		self.config_manager = ConfigurationManager()
		t0 = datetime.datetime.now()
		self.peer_size = self.config_manager.get_number_of_peers()
		self.topics_num = self.config_manager.get_number_of_topics()
		self.min_sim_threshold = self.config_manager.get_min_similarity_threshold()
		self.max_sim_threshold = self.config_manager.get_max_similarity_threshold()
		self.first_user =  self.config_manager.get_first_user()
		self.last_user = self.config_manager.get_last_user()
		paper_presentation = self.config_manager.get_paper_presentation()
		print("*** PARSING DATA ***")
		self.parser = DataParser(self.DATASET, self.topics_num, paper_presentation)
		self.sampling_method = self.config_manager.get_sampling()
		print("*** DATA PARSED ***")
		# print("*** EXTRACTING KEYWORDS ***")
		# labels, data = self.parser.get_raw_data()
		# self.keyword_extractor = KeywordExtractor(labels, data, self.DATASET)
		# self.documents_matrix = self.keyword_extractor.get_tfidf()
		self.documents_matrix = self.parser.get_document_word_distribution()
		print("*** KEYWORDS EXTRACTED ***")
		self.ratings = self.parser.get_ratings_matrix()
		self.k_folds = 5
		print("TRACE")
		r = self.ratings[0].nonzero()[0]
		print(self.ratings[0].nonzero()[0])
		a = self.documents_matrix[r[0]]
		print(self.documents_matrix[0].nonzero()[0])
		print(self.documents_matrix[1].nonzero()[0])
		self.train_indices, self.test_indices = self.get_kfold_indices()
		self.train()
		print("Done in {}".format(datetime.datetime.now() - t0))

	def get_test_documents(self, test_indices, user):
		documents = []
		indices = []
		for index in test_indices[user]:
			documents.append(self.documents_matrix[index])
			indices.append(index)
		return np.array(documents), np.array(indices)

	def train(self):

		print("*** TRAINING ***")
		ndcgs = []
		mrrs = []
		recalls_10 = []
		recalls_50 = []
		recalls_100 = []
		recalls_200 = []
		folds_results = []
		for fold in range(self.k_folds):
			self.fold_train_indices, self.fold_test_indices = self.get_fold_indices(fold, self.train_indices, self.test_indices)
			self.train_data, self.test_data = self.get_fold(fold, self.train_indices, self.test_indices)
			## TODO Look at the users that have p+ p+ as pair
			self.peer_extractor = PeerExtractor(self.train_data, self.documents_matrix, self.sampling_method, 'cosine', self.peer_size, self.min_sim_threshold)
			self.similarity_matrix = self.peer_extractor.get_similarity_matrix()
			fold_results = []
			for user in range(self.ratings.shape[0]):
				t0 = datetime.datetime.now()
				pairs = self.peer_extractor.get_user_peer_papers(user)
				feature_vectors = []
				labels = []
				print("*** BUILDNG PAIRS ***")
				i = 0
				for pair in pairs:
					# self.parser.get_author_similarity(pair[1], self.ratings[user].nonzero())
					feature_vector, label = self.build_vector_label_sim_svm(pair, user)
					feature_vectors.append(feature_vector[0])
					feature_vectors.append(feature_vector[1])
					labels.append(label[0])
					labels.append(label[1])
					i += 1
				print("*** PAIRS BUILT ***")
				feature_vectors = np.array(feature_vectors)
				print("*** FITTING SVR MODEL ***")

				# This classifier supports nonleniar models, applies a kernel, uses libsvm implementation,
				# clf = SKSVR(verbose=True)

				# print("Vectors size are {}".format(feature_vectors.shape))

				# Using grid search for fitting hyper barameters (Penalty, C)
				tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 1, 10, 100, 1000]}]
				#grid_clf = GridSearchCV(LinearSVC(dual=False, tol=0.0001, random_state=41), tuned_parameters, cv=3, scoring='recall', n_jobs=-1)
				grid_clf = SKSVR(verbose=True)
				grid_clf.fit(feature_vectors, labels)
				# print("Best parameters set found on development set: {}").format(grid_clf.best_estimator_)

				# Using LinnearSVC instead of SVC, it uses the implementation of liblinear, should be faster for linear models.
				# Setting the classifier with the best parameters found in gridsearch
				# clf = LinearSVC(penalty=grid_clf.best_params_['penalty'], loss='squared_hinge', dual=False,C=grid_clf.best_params_['C'], random_state=41)
				# Learning the model


				print("*** FITTED SVR FOR USER {} ***".format(user))
				test_documents, test_indices = self.get_test_documents(self.fold_test_indices, user)
				predictions = grid_clf.decision_function(test_documents)
				ndcg_at_10, mrr_at_10 = self.evaluate(user, predictions, test_indices, 10)
				recall_at_10 = self.calculate_top_recall(user, predictions, test_indices, 10)
				recall_at_50 = self.calculate_top_recall(user, predictions, test_indices, 50)
				recall_at_100 = self.calculate_top_recall(user, predictions, test_indices, 100)
				recall_at_200 = self.calculate_top_recall(user, predictions, test_indices, 200)
				fold_results.append([ndcg_at_10, mrr_at_10, recall_at_10, recall_at_50, recall_at_100, recall_at_200])
				ndcgs.append(ndcg_at_10)
				mrrs.append(mrr_at_10)
				recalls_10.append(recall_at_10)
				recalls_50.append(recall_at_50)
				recalls_100.append(recall_at_100)
				recalls_200.append(recall_at_200)
				print("Fold {}, for user {}, training {}, test {}, train_pos {}, test_pos {} :".format(fold, user, feature_vectors.shape, len(test_documents), sum(self.train_data[user]), sum(self.test_data[user])))
				print(str(['{:^7}'.format(v) for v in ["NDCG@10", "MRR", "REC@10", "REC@50", "REC@100", "REC@200"]]))
				print(str(['{:06.5f}'.format(v) for v in [ndcg_at_10, mrr_at_10, recall_at_10, recall_at_50, recall_at_100, recall_at_200]]))
				print('Average so far:')
				print(str(['{:06.5f}'.format(v) for v in
						   [np.array(ndcgs).mean(), np.array(mrrs).mean(), np.array(recalls_10).mean(), np.array(recalls_50).mean(), np.array(recalls_100).mean(), np.array(recalls_200).mean()]]))
				print("user {} took {}".format(user, datetime.datetime.now() - t0))
			print("Averages for fold {}:").format(fold)
			print(str(['{:^7}'.format(v) for v in ["NDCG@10", "MRR", "REC@10", "REC@50", "REC@100", "REC@200"]]))
			print(str(['{:06.5f}'.format(v) for v in np.array(fold_results).mean(axis = 0)]))

			folds_results.append(np.array(fold_results).mean(axis = 0))
		print("-----------------------------------------------------------")
		print("-----------------------------------------------------------")
		print("Final results for peer size {} and {} topics and Sim_threshold = {}:".format(self.peer_size, self.topics_num, self.min_sim_threshold))
		print('                '+str(['{:^7}'.format(v) for v in ["NDCG@10", "MRR", "REC@10", "REC@50", "REC@100", "REC@200"]]))
		for (i,f) in enumerate(folds_results):
			print('For fold {}:     '.format(i)+str(['{:06.5f}'.format(v) for v in f]))
		print('Total Averages: '+str(['{:06.5f}'.format(v) for v in [np.array(ndcgs).mean(), np.array(mrrs).mean(),
																	 np.array(recalls_10).mean(),
																	 np.array(recalls_50).mean(),
																	 np.array(recalls_100).mean(),
																	 np.array(recalls_200).mean()]]))

	def evaluate(self, user, predictions, test_indices, k):
		dcg = 0.0
		idcg = 0.0
		mrr = 0.0
		ndcgs = []
		top_predictions = TopRecommendations(k)
		for prediction, index in zip(predictions, test_indices):
			top_predictions.insert(index, prediction)
		recommendation_indices = top_predictions.get_indices()
		for pos_index, index in enumerate(recommendation_indices):
			hit_found = False
			dcg += (self.test_data[user][index] / np.log2(pos_index + 2))
			idcg += 1 / np.log2(pos_index + 2)
			if self.test_data[user][index] == 1 and mrr == 0.0:
				mrr = 1.0 / (pos_index + 1) * 1.0
			if pos_index + 1 == k:
				break
		if idcg != 0:
			return (dcg / idcg), mrr
		return 0, mrr

	def calculate_top_recall(self, user, predictions, test_indices, k):
		recall = 0.0
		top_predictions = TopRecommendations(k)
		for prediction, index in zip(predictions, test_indices):
			top_predictions.insert(index, prediction)
		nonzeros = self.test_data[user].nonzero()[0]
		denom = len(nonzeros) * 1.0
		for index in top_predictions.get_indices():
			if index in nonzeros:
				recall += 1.0
		if recall == 0:
			return 0
		return recall / min(denom, k)



	def get_user_paper_similarity(self, user, paper):
		## TODO: BUG fixing, liked_papers must be positives from the training set, not from all of the rarings!
		liked_papers = self.ratings[user].nonzero()[0]
		return self.similarity_matrix[paper][liked_papers].max()

	def build_vector_label_conf_svr(self, pair, user):
		pivot = pair[0]
		peer = pair[1]
		feature_vector = []
		label = []
		feature_vector.append((self.documents_matrix[pivot] - self.documents_matrix[peer]) * self.get_confidence(user, peer))
		label.append(1 - self.similarity_matrix[pivot][peer])
		feature_vector.append((self.documents_matrix[peer] - self.documents_matrix[pivot]) * self.get_confidence(user, peer))
		label.append(1 - self.similarity_matrix[pivot][peer])
		# feature_vector = (self.documents_matrix[pivot] - self.documents_matrix[peer]) * self.get_user_paper_similarity(user, peer)
		# label = self.similarity_matrix[pivot][peer]
		return feature_vector, label

	def build_vector_label_conf_svm(self, pair, user):
		"""
        Builds two feature vectors for each pair (p1, p2) as:
        confidence(p2) * (p1-p2) -> +1
        confidence(p2) * (p2-p1) -> -1
        """
		pivot = pair[0]
		peer = pair[1]
		feature_vector = []
		label = []
		feature_vector.append(
			(self.documents_matrix[pivot] - self.documents_matrix[peer]) * self.get_confidence(user, peer))
		label.append(1)
		feature_vector.append(
			(self.documents_matrix[peer] - self.documents_matrix[pivot]) * self.get_confidence(user, peer))
		label.append(-1)
		# feature_vector = (self.documents_matrix[pivot] - self.documents_matrix[peer]) * self.get_user_paper_similarity(user, peer)
		# label = self.similarity_matrix[pivot][peer]
		return feature_vector, label

	def build_vector_label_sim_svm(self, pair, user):
		"""
		Builds two feature vectors for each pair (p1, p2) as:
		(1-user_paper_similarity(user, p2)) * (p1-p2) -> +1
		(1-user_paper_similarity(user, p2)) * (p2-p1) -> -1
		"""
		pivot = pair[0]
		peer = pair[1]
		feature_vector = []
		label = []
		feature_vector.append((self.documents_matrix[pivot] - self.documents_matrix[peer]) * (1- self.get_user_paper_similarity(user, peer)))
		label.append(1)
		feature_vector.append((self.documents_matrix[peer] - self.documents_matrix[pivot]) * (1-self.get_user_paper_similarity(user, peer)))
		label.append(-1)
		return feature_vector, label

	def get_confidence(self, user, paper):
		user_papers = self.train_data[user].nonzero()[0]
		author_similarity = self.parser.get_author_similarity(paper, user_papers)
		conference_similarity = self.parser.get_conference_similarity(paper, user_papers)
		textual_similarity = self.get_user_paper_similarity(user, paper)
		textual_similarity = (textual_similarity * 2) - 1
		author_similarity += 1
		conference_similarity += 1
		return (0.5 - ((1/8) * (conference_similarity * author_similarity * textual_similarity)))

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