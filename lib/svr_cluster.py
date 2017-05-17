import os
import sys
from util.data_parser import DataParser
from sklearn.svm import LinearSVC as SKSVR
from util.data_dumper import DataDumper
from lib.peer_extractor import PeerExtractor
import multiprocessing
from functools import partial
import numpy as np
import datetime
from util.top_similar import TopSimilar as TopRecommendations



def _pickle_method(method):
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
		cls_name = cls.__name__.lstrip('_')
		func_name = '_' + cls_name + func_name
	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	for cls in cls.__mro__:
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class SVRCluster(object):

	DATASET = 'dummy'

	def __init__(self, start_id, end_id):
		print(start_id)
		self.start_id = start_id
		self.end_id = end_id
		self.sim_threshold = 0.07
		self.paper_presentation = "keywords"
		self.peer_size = 20
		self.parser = DataParser(self.DATASET, 20, self.paper_presentation)
		self.documents_matrix = self.parser.get_document_word_distribution()
		self.ratings = self.parser.get_ratings_matrix()
		self.k_folds = 5
		loader = DataDumper(self.DATASET, 'splits')
		_, self.train_indices = (loader.load_matrix('train_indices'))
		_, self.test_indices = (loader.load_matrix('test_indices'))

		#num_cores = multiprocessing.cpu_count()
		print("Creating parallel job")
		p = multiprocessing.Pool()
		#self.go()
		results = p.map(self.recommend_user, range(start_id, end_id))
		#results = Parallel(n_jobs=num_cores)(delayed(self.recommend_user)(i) for i in range(start_id, end_id))
		path = self._create_path('results', 'metrics-results.csv')
		np.savetxt(path, np.array(results), delimiter=",")


	def _create_path(self, folder, matrix_name):
		"""
		Function creates a string uniquely representing the matrix it also
		uses the config to generate the name.

		:param str matrix_name: Name of the matrix.
		:param int n_rows: Number of rows of the matrix.
		:returns: A string representing the matrix path.
		:rtype: str
		"""
		path = matrix_name
		base_dir = os.path.dirname(os.path.realpath(__file__))
		return os.path.join(os.path.dirname(base_dir), folder, path)


	def recommend_user(self, user):
		print("Calculating for user {}".format(user))
		for fold in range(self.k_folds):
			self.fold_train_indices, self.fold_test_indices = self.get_fold_indices(fold, self.train_indices, self.test_indices)
			train_data, test_data = self.get_fold(fold, self.train_indices, self.test_indices)
			peer_extractor = PeerExtractor(train_data, self.documents_matrix, "least_similar_k", 'cosine', self.peer_size, self.sim_threshold)
			self.similarity_matrix = peer_extractor.get_similarity_matrix()
			t0 = datetime.datetime.now()
			pairs = peer_extractor.get_user_peer_papers(user)
			feature_vectors = []
			labels = []

			print("*** BUILDNG PAIRS ***")
			i = 0
			for pair in pairs:
				feature_vector, label = self.build_vector_label_sim_svm(pair, user)
				feature_vectors.append(feature_vector[0])
				feature_vectors.append(feature_vector[1])
				labels.append(label[0])
				labels.append(label[1])
				i += 1
			print("*** PAIRS BUILT ***")
			feature_vectors = np.array(feature_vectors)
			print("*** FITTING SVR MODEL ***")
			grid_clf = SKSVR(verbose=True)
			grid_clf.fit(feature_vectors, labels)
			results = []
			test_documents, test_indices = self.get_test_documents(self.fold_test_indices, user)
			predictions = grid_clf.decision_function(test_documents)
			ndcg_at_10, mrr_at_10 = self.evaluate(user, predictions, test_indices, 10, test_data)
			results.append(ndcg_at_10)
			results.append(mrr_at_10)

			recall_xs = [10, 50, 100, 200]
			for recall_x in recall_xs:
				results.append(self.calculate_top_recall(user, predictions, test_indices, recall_x, test_data))
		print results
		return results


	def evaluate(self, user, predictions, test_indices, k, test_data):
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
			dcg += test_data[user][index] / np.log2(pos_index + 2)
			idcg += 1 / np.log2(pos_index + 2)
			if test_data[user][index] == 1 and mrr == 0.0:
				mrr = 1.0 / (pos_index + 1) * 1.0
			if pos_index + 1 == k:
				break
		if idcg != 0:
			return (dcg / idcg), mrr
		return 0, mrr

	def calculate_top_recall(self, user, predictions, test_indices, k, test_data):
		recall = 0.0
		top_predictions = TopRecommendations(k)
		for prediction, index in zip(predictions, test_indices):
			top_predictions.insert(index, prediction)
		nonzeros = test_data[user].nonzero()[0]
		denom = len(nonzeros) * 1.0
		for index in top_predictions.get_indices():
			if index in nonzeros:
				recall += 1.0
		if recall == 0:
			return 0
		return recall / min(denom, k)


	def get_test_documents(self, test_indices, user):
		documents = []
		indices = []
		for index in test_indices[user]:
			documents.append(self.documents_matrix[index])
			indices.append(index)
		return np.array(documents), np.array(indices)

	def get_user_paper_similarity(self, user, paper):
		liked_papers = self.ratings[user].nonzero()[0]
		return self.similarity_matrix[paper][liked_papers].max()

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
		index = fold_num
		for ctr in range(self.ratings.shape[0]):
			current_train_fold_indices.append(self.train_indices[index])
			current_test_fold_indices.append(self.test_indices[index])
			index += self.k_folds
		return (current_train_fold_indices, current_test_fold_indices)

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
		index = fold_num
		for ctr in range(self.ratings.shape[0]):
			current_train_fold_indices.append(self.train_indices[index])
			current_test_fold_indices.append(self.test_indices[index])
			index += self.k_folds
		return self.generate_kfold_matrix(current_train_fold_indices, current_test_fold_indices)

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
