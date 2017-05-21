#!/usr/bin/env python
"""
This module provides functionalities for extracting peer papers
"""
import numpy as np
import random
from util.top_similar import TopSimilar



def get_user_peer_papers(user, peer_extraction_method, ratings, sim_min_threshold, sim_max_threshold, sim_threshold, peer_size):

	if peer_extraction_method == 'random':
		return get_random_peer_papers(user)
	elif peer_extraction_method == 'least_k':
			return get_least_k(user)
	elif peer_extraction_method =="least_similar_k":
			return get_least_similar_k(user, ratings, sim_min_threshold, sim_max_threshold, sim_threshold, peer_size)

def get_least_k(self, user):
	## TODO Add it
	pass

def get_least_similar_k(user, ratings, similarity_matrix, sim_min_threshold, sim_max_threshold, peer_size):
	## Randomize
	positive_papers = ratings[user].nonzero()[0]
	pairs = []
	for paper in positive_papers:
		top_similar = TopSimilar(peer_size)
		## Get papers with non zero similarity
		# nonzeros = self.similarity_matrix[paper].nonzero()[0]
		nonzeros = np.where((similarity_matrix[paper] > sim_min_threshold)&(similarity_matrix[paper] < sim_max_threshold))[0]
		for index in nonzeros:
			if paper == index:
				continue
			# This is a bad bug, the first index should be paper not user!
			# top_similar.insert(index, 1 - self.similarity_matrix[user][index])
			top_similar.insert(index, 1 - similarity_matrix[paper][index])
		similar_papers = top_similar.get_indices()
		for similar_paper in similar_papers:
			# Get the similarity between the peer paper and the user profile
			peer_user_similarity = similarity_matrix[similar_paper][positive_papers].max()
			pairs.append((paper, similar_paper,peer_user_similarity))
	return pairs





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

