#!/usr/bin/env python
"""
This module provides functionalities for extracting peer papers
"""
import numpy as np
import random
from scipy import sparse
from util.top_similar import TopSimilar



def get_user_peer_papers(peer_extraction_method, user_ratings, sim_min_threshold, sim_max_threshold, sim_threshold, peer_size):

	#if peer_extraction_method == 'random':
	#	return get_random_peer_papers(user)
	#elif peer_extraction_method == 'least_k':
	#		return get_least_k(user)
	#elif peer_extraction_method =="least_similar_k":
			return get_least_similar_k(user_ratings, sim_min_threshold, sim_max_threshold, sim_threshold, peer_size)

def get_least_k(self, user):
	## TODO Add it
	pass

def get_least_similar_k(user_ratings, similarity_matrix, sim_min_threshold, sim_max_threshold, peer_size):
	## Randomize
	positive_papers = user_ratings.nonzero()[0]
	relevant_papers = []
	peer_papers = []
	scores = []
	for paper in positive_papers:
		peers_queue = TopSimilar(peer_size)
		## Get papers with non zero similarity
		if sparse.issparse(similarity_matrix):  # For Sparse similarity matrix, the matrix is Compressed Sparse Row matrix:  csr_matrix
			for index in similarity_matrix[paper].nonzero()[1]:
				if similarity_matrix[paper, index] >= sim_min_threshold and similarity_matrix[paper, index] <= sim_max_threshold:
					if paper == index:
						continue
					peers_queue.insert(index, 1 - similarity_matrix[paper, index])
			peers_indices = peers_queue.get_indices()
			for peer in peers_indices:
				# Get the similarity between the peer paper and the user profile
				peer_user_similarity = similarity_matrix[peer,positive_papers].max()
				relevant_papers.append(paper)
				peer_papers.append(peer)
				scores.append(peer_user_similarity)
		else:  # The following row is for dense similarity matrix:
			nonzeros = np.where((similarity_matrix[paper] >= sim_min_threshold)&(similarity_matrix[paper] <= sim_max_threshold))[0]
			for index in nonzeros:
				if paper == index:
					continue
				# This is a bad bug, the first index should be paper not user!
				# top_similar.insert(index, 1 - self.similarity_matrix[user][index])
				peers_queue.insert(index, 1 - similarity_matrix[paper][index])
			peers_indices = peers_queue.get_indices()
			for peer in peers_indices:
				# Get the similarity between the peer paper and the user profile
				peer_user_similarity = similarity_matrix[peer][positive_papers].max()
				relevant_papers.append(paper)
				peer_papers.append(peer)
				scores.append(peer_user_similarity)
	return (relevant_papers, peer_papers, scores)





def get_random_peer_papers(self, user):

	if user in self.pairs:
		return self.pairs[user]

	positive_papers = self.ratings[user].nonzero()[0]
	negative_papers = np.where(self.ratings[user] == 0)[0]
	pairs = []
	for paper in positive_papers:
		random_indices = random.sample(range(0, len(negative_papers)), self.k)
		for index in random_indices:
			pairs.append((paper, negative_papers[index]))
	return pairs


def get_textual_similarity(self, user, paper):
	liked_papers = self.ratings[user].nonzero()
	return self.similarity_matrix[paper][liked_papers].max()



def build_pair_vector_label_user_sim(pair, document_matrix_shared):
	"""
	Builds two feature vectors for each pair (p1, p2) as:
	(1-user_paper_similarity(user, p2)) * (p1-p2) -> +1
	(1-user_paper_similarity(user, p2)) * (p2-p1) -> -1
	"""
	pivot = pair[0]
	peer = pair[1]
	peer_user_sim = pair[2]
	feature_vector = []
	label = []
	feature_vector.append((document_matrix_shared[pivot] - document_matrix_shared[peer]) * (1- peer_user_sim))
	label.append(1)
	feature_vector.append((document_matrix_shared[peer] - document_matrix_shared[pivot]) * (1- peer_user_sim))
	label.append(-1)
	return feature_vector, label

