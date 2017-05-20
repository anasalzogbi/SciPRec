#!/usr/bin/env python

import sys
import getopt
import os
import sys
from util.data_parser import DataParser
from sklearn.svm import LinearSVC as SKSVR
from util.data_dumper import DataDumper
from lib.peer_extractor import PeerExtractor
import multiprocessing
import numpy as np
import datetime
from util.top_similar import TopSimilar as TopRecommendations




def create_path(self, folder, matrix_name):
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


def recommend_user(user):
	print("Calculating for user {}".format(user))
	for fold in range(k_folds):
		fold_train_indices, fold_test_indices = get_fold_indices(fold, train_indices_shared, test_indices_shared, ratings_shared, k_folds)

		train_data, test_data = get_fold(fold, train_indices_shared, test_indices_shared, ratings_shared, k_folds)
		peer_extractor = PeerExtractor(train_data, document_matrix_shared, "least_similar_k", 'cosine', peer_size, sim_threshold)
		similarity_matrix = peer_extractor.get_similarity_matrix()
		t0 = datetime.datetime.now()
		pairs = peer_extractor.get_user_peer_papers(user)
		feature_vectors = []
		labels = []

		print("*** BUILDNG PAIRS ***")
		i = 0
		for pair in pairs:
			feature_vector, label = build_vector_label_sim_svm(pair, user, document_matrix_shared, ratings_shared, similarity_matrix)
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
		test_documents, test_indices = get_test_documents(fold_test_indices, user, document_matrix_shared)
		predictions = grid_clf.decision_function(test_documents)
		ndcg_at_10, mrr_at_10 = evaluate(user, predictions, test_indices, 10, test_data)
		results.append(ndcg_at_10)
		results.append(mrr_at_10)

		recall_xs = [10, 50, 100, 200]
		for recall_x in recall_xs:
			results.append(calculate_top_recall(user, predictions, test_indices, recall_x, test_data))
	print results
	return results


def evaluate(user, predictions, test_indices, k, test_data):
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

def calculate_top_recall(user, predictions, test_indices, k, test_data):
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


def get_test_documents(test_indices, user, document_matrix_shared):
	documents = []
	indices = []
	for index in test_indices[user]:
		documents.append(document_matrix_shared[index])
		indices.append(index)
	return np.array(documents), np.array(indices)

def get_user_paper_similarity(user, paper, ratings_shared, similarity_matrix_shared):
	liked_papers = ratings_shared[user].nonzero()[0]
	return similarity_matrix_shared[paper][liked_papers].max()

def build_vector_label_sim_svm(pair, user, document_matrix_shared, ratings_shared, similarity_matrix_shared):
	"""
	Builds two feature vectors for each pair (p1, p2) as:
	(1-user_paper_similarity(user, p2)) * (p1-p2) -> +1
	(1-user_paper_similarity(user, p2)) * (p2-p1) -> -1
	"""
	pivot = pair[0]
	peer = pair[1]
	feature_vector = []
	label = []
	feature_vector.append((document_matrix_shared[pivot] - document_matrix_shared[peer]) * (1- get_user_paper_similarity(user, peer, ratings_shared, similarity_matrix_shared)))
	label.append(1)
	feature_vector.append((document_matrix_shared[peer] - document_matrix_shared[pivot]) * (1- get_user_paper_similarity(user, peer, ratings_shared, similarity_matrix_shared)))
	label.append(-1)
	return feature_vector, label

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

def get_fold(fold_num, train_indices_shared, test_indices_shared, ratings_shared, k_folds):
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
	return generate_kfold_matrix(current_train_fold_indices, current_test_fold_indices, ratings_shared)

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





# dp = DataParser('citeulike-a')
# labels, data = dp.get_raw_data()
# e = KeywordExtractor(labels, data)
# tf_idf = e.tf_idf
# peer_extractor = PeerExtractor(dp.ratings, tf_idf, 'least_k', 'cosine', 10)
start_id = -1
end_id = -1
sim_threshold = 0.07
paper_presentation = "keywords"
peer_size = 20
DATASET = 'dummy'

argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv,"hs:e:",["start_id=","end_id="])
except getopt.GetoptError:
	print 'runnables.py -s <start_id> -e <end_id>'
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print 'runnables.py -s <start_id> -e <end_id>'
		sys.exit()
	elif opt in ("-s", "--start_id"):
		start_id = arg
	elif opt in ("-e", "--end_id"):
		end_id = arg
print("Starting user: {}".format(start_id))
print("Ending user: {}".format(end_id))

start_id = int(start_id)
end_id = int(end_id)

parser = DataParser(DATASET, 20, paper_presentation)
document_matrix_shared = parser.get_document_word_distribution()
ratings_shared = parser.get_ratings_matrix()
k_folds = 5
loader = DataDumper(DATASET, 'splits')
_, train_indices_shared = (loader.load_matrix('train_indices'))
_, test_indices_shared = (loader.load_matrix('test_indices'))

#num_cores = multiprocessing.cpu_count()

print("Creating parallel job")
p = multiprocessing.Pool()
#self.go()
results = p.map(recommend_user, range(start_id, end_id))
p.close()
p.join()
#results = Parallel(n_jobs=num_cores)(delayed(self.recommend_user)(i) for i in range(start_id, end_id))
path = create_path('results', 'metrics-results.csv')
np.savetxt(path, np.array(results), delimiter=",")

