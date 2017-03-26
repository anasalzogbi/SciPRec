#!/usr/bin/env python
"""
This module provides functionalities for extracting peer papers
"""
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from util.top_similar import TopSimilar
import sys

class PeerExtractor(object):


	def __init__(self, ratings, documents, method, similarity_metric, k):
		print("*** PEER EXTRACTOR INIT ***")
		self.ratings = ratings
		self.method = method
		self.k = k
		self.documents = documents
		self.similarity_metric = similarity_metric
		self.pairs = {}
		self.similarity_matrix = None
		print("*** Calculating Similarity ***")
		self.calculate_pairwise_similarity()

	def get_similarity_matrix(self):
		return self.similarity_matrix

	def get_user_peer_papers(self, user):

		if self.method == 'random':
			self.pairs[user] = self.get_random_peer_papers(user)
		else:
			if self.method == 'least_k':
				self.pairs[user] = self.get_least_k(user)
			else: 
				self.pairs[user] = self.get_least_similar_k(user)

		return self.pairs[user]

	def get_least_k(self, user):
		## TODO Add it
		pass

	def get_least_similar_k(self, user):
		## Randomize
		positive_papers = self.ratings[user].nonzero()[0]
		negative_papers = np.where(self.ratings[user] == 0)[0]
		user_ratings = self.ratings[user]
		top_similar = TopSimilar(self.k)
		pairs = []
		for paper in positive_papers:
			top_similar = TopSimilar(self.k)
			## Get papers with non zero similarity
			nonzeros = self.similarity_matrix[paper].nonzero()[0]
			for index in nonzeros:
				if paper == index:
					continue
				top_similar.insert(index, 1 - self.similarity_matrix[user][index])
			similar_papers = top_similar.get_indices()
			for similar_paper in similar_papers:
				pairs.append((paper, similar_paper))
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

	def calculate_pairwise_similarity(self):
		if self.similarity_matrix is not None:
			return
		docs_count = self.ratings.shape[1]
		self.similarity_matrix = np.eye(docs_count)
		if self.similarity_metric == 'dot':
			# Compute pairwise dot product
			for i in range(docs_count):
				for j in range(i, docs_count):
					self.similarity_matrix[i][j] = self.documents[i].dot(self.documents[j].T)
					self.similarity_matrix[j][i] = self.similarity_matrix[i][j]

		if self.similarity_metric == 'cosine':
			self.similarity_matrix = cosine_similarity(sparse.csr_matrix(self.documents))
		#self.similarity_matrix[self.similarity_matrix == 1.0] = 0
		similarity_matrix = self.documents.dot(self.documents.T)

	def get_textual_similarity(self, user, paper):
		liked_papers = self.ratings[user].nonzero()
		return self.similarity_matrix[paper][liked_papers].max()

