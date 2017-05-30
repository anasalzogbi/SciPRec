#!/usr/bin/env python
"""
This module provides functionalities for parsing the data
"""

import os
import csv
import numpy as np
import datetime
from sklearn.decomposition import LatentDirichletAllocation


class DataParser(object):

	def __init__(self, dataset, topics_num, papers_presentation):
		"""
		Initializes the data parser given a dataset name
		"""
		self.base_folder = '../datasets/Extended_ctr'
		self.dataset = dataset
		self.papers_presentation = papers_presentation

		if dataset == 'citeulike-a':
			self.dataset_folder = self.base_folder + '/citeulike_a_extended'
		elif dataset == 'citeulike-t':
			self.dataset_folder = self.base_folder + '/citeulike-t_extended'
		elif dataset == 'dummy':
			self.dataset_folder = self.base_folder + '/dummy'
		else:
			print("Warning: Given dataset not known, setting to citeulike-a")
			self.dataset_folder = self.base_folder + '/citeulike_a_extended'

		self.paper_count = None
		self.user_count = None
		self.words = {}
		self.words_count = 0
		self.id_map = {}
		self.authors_map = {}
		self.topics_num = topics_num
		self.process()
	

	def process(self):
		"""
		Starts parsing the data and gets matrices ready for training
		"""
		#self.raw_labels, self.raw_data = self.parse_paper_raw_data()

		self.ratings = self.generate_ratings_matrix()
		self.build_document_word_matrix()
		#print("shape")
		#print(self.document_words.shape)
		if self.papers_presentation == 'lda':
			print ('***** LDA- topics {}'.format(self.topics_num))
			lda = LatentDirichletAllocation(n_topics=self.topics_num, max_iter=10,
											learning_method='online',
											learning_offset=50., random_state=0,
											verbose=0, n_jobs=-1)
			self.document_distribution = lda.fit_transform(self.document_words)
		if self.papers_presentation == 'attributes':
			self.feature_labels, self.feature_matrix = self.parse_paper_features()
		self.parse_authors()

	def build_document_word_matrix(self):
		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'mult.dat')
		documents_hash = {}
		docs_count = 0
		words_count = 1
		with open(path, "r") as f:
			for doc_id, line in enumerate(f):
				splitted = line.split(" ")
				count = int(splitted[0])
				words = []
				for i in range(count):
					word_id_count = splitted[i+1].split(":")
					word_id = int(word_id_count[0])
					count = int(word_id_count[1])
					words.append((word_id, count))
					words_count = max(word_id + 1, words_count)
				documents_hash[doc_id] = words
				docs_count += 1
		document_words = np.zeros((docs_count, words_count))
		for key, entry in documents_hash.iteritems():
			for value in entry:
				word_id = value[0]
				word_count = value[1]
				document_words[key][word_id] = word_count
		del documents_hash
		print ("**** Document Matrix: {}".format(document_words.shape))
		self.document_words = document_words


	def get_document_word_distribution(self):
		if self.papers_presentation == 'lda':
			return self.document_distribution
		elif self.papers_presentation == 'attributes':
			return self.feature_matrix
		return self.document_words


	def parse_paper_features(self):
		"""
		Parses paper features
		"""
		now = datetime.datetime.now()
		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'paper_info.csv')
		with open(path, "r") as f:			
			reader = csv.reader(f, delimiter='\t')
			first_line = True
			feature_vec = []
			i = 0
			row_length = 0
			labels_ids = []
			for line in reader:
				if first_line:
					labels = ["type", "publisher", "year", "address", "booktitle", "journal", "series"]
					for j, entry in enumerate(line):
						if entry in labels:
							labels_ids.append(j)
					row_length = len(labels_ids)
					first_line = False
					i += 1
					continue
				paper_id = line[0]
				if int(paper_id) != i:
					for _ in range(int(paper_id) - i):
						feature_vec.append([None] * row_length)
						i += 1
				current_entry = []
				for k, label_id in enumerate(labels_ids):
					if k == 5:
						current_entry.append(now.year - int(line[label_id]))
					else:
						current_entry.append(line[label_id])
				feature_vec.append(current_entry)
				i += 1
		
		if self.paper_count is None:
			self.paper_count = len(feature_vec)
		return labels, np.array(feature_vec)

	def insert_word(self, word):
		if word in self.words:
			return
		self.words[word] = 1
		self.words_count += 1

	def get_paper_conference(self, paper):
		journal_index = -1
		booktitle_index = -1
		series_index = -1
		for index, label in enumerate(self.feature_labels):
			if label == 'journal':
				journal_index = index
			if label == 'booktitle':
				booktitle_index = index
			if label == 'series':
				series_index = index
		if self.feature_matrix[paper][journal_index] != "\N":
			return self.feature_matrix[paper][journal_index]
		if self.feature_matrix[paper][booktitle_index] != "\N":
			return self.feature_matrix[path][booktitle_index]
		if self.feature_matrix[paper][series_index] != "\N":
			return self.feature_matrix[path][series_index]
		return None
	def parse_authors(self):
		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'authors.csv')
		with open(path, "r") as f:
			reader = csv.reader(f, delimiter='\t')
			first_line = True
			for line in reader:
				if first_line:
					first_line = False
					continue
				key = self.id_map[int(line[0])]
				if key in self.authors_map:
					if self.authors_map[key] is not None:
						self.authors_map[key] = self.authors_map[key].append(line[1])
					else:
						self.authors_map[key] = [line[1]]
				else:
					self.authors_map[key] = [line[1]]

	def get_author_similarity(self, paper, user_papers):
		print(type(self.authors_map))
		sim = 0
		paper_author = self.authors_map[paper]
		user_authors = []
		for user_paper in user_papers:
			if self.authors_map[user_paper] is not None:
				user_authors.append(self.authors_map[user_paper])
		intersection = [filter(lambda x: x in paper_author, sublist) for sublist in user_authors]
		sim = sum(len(x) for x in intersection)
		return sim / sum(len(x) for x in user_authors) * 1.0

	def get_conference_similarity(self, paper, user_papers):
		paper_conference = self.get_paper_conference(paper)
		denom = len(user_papers) * 1.0
		numer = 0.0
		for user_paper in user_papers:
			user_conference = self.get_paper_conference(user_paper)
			if user_conference == paper_conference:
				numer += 1.0
		return numer / denom 

	def parse_paper_raw_data(self):
		"""
		Parses paper raw data
		"""
		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'raw-data.csv')
		delimiter = ','
		total = 0
		if self.dataset == 'citeulike-t':
			delimiter = '\t'
		with open(path, "r") as f:
			reader = csv.reader(f, delimiter=delimiter)
			first_line = True
			data_vec = []
			row_length = 0
			for doc_id, line in enumerate(reader):
				if first_line:
					labels = line[1:]
					row_length = len(line)
					first_line = False
					continue
				data_vec.append(line[1:])
				for word in line[3].split(" "):
					self.insert_word(word)
				for word in line[4].split(" "):
					self.insert_word(word)
				self.id_map[int(float(line[2]))] = doc_id - 1


		if self.paper_count is None:
			self.paper_count = len(data_vec)

		#print "Total is "
		#print(self.words_count)
		return labels, np.array(data_vec)

	def generate_ratings_matrix(self):
		"""
		Generates a rating matrix of user x paper
		"""
		if self.paper_count is None:
			self.raw_labels, self.raw_data = self.parse_paper_raw_data()

		#print self.paper_count
		path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dataset_folder, 'users.dat')
		self.user_count = sum(1 for line in open(path))
		ratings = np.zeros((self.user_count, self.paper_count))
		i = 0
		with open(path, "r") as f:
			for line in f:
				splitted = line.replace("\n", "").split(" ")
				for paper_id in splitted:
					ratings[i][int(paper_id)] = 1
				
				i += 1

		return ratings

	def get_raw_data(self):
		return self.raw_labels, self.raw_data

	def get_feature_vector(self):
		return self.feature_labels, self.feature_matrix

	def get_ratings_matrix(self):
		return self.ratings
				



