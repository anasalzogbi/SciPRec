#!/usr/bin/env python
"""
This module provides simple wrappign functions for topia
"""
from topia.termextract import extract
from data_parser import DataParser
from util.data_dumper import DataDumper
import numpy as np
import traceback


class KeywordExtractor(object):

	def __init__(self, raw_labels, raw_data, dataset):
		self.dataset = dataset
		self.raw_labels = raw_labels
		self.raw_data = raw_data
		self.document_count = len(raw_data)
		print("*** TRYING TO LOAD TFIDF MATRIX ***")
		data_dumper = DataDumper(dataset)
		found, self.tf_idf = data_dumper.load_matrix()
		found = False
		if found is False:
			self.word_count = 0
			self.words = {}
			self.words_inverted = {}
			documents_to_words = self.extract_word_distribution()
			self.term_frequency = self.build_document_word_matrix(documents_to_words)
			self.tf_idf = self.term_frequency * self.calculate_idf()

			#data_dumper.save_matrix(self.tf_idf)
		self.analyze_frequencies(self.term_frequency)


	def calculate_idf(self):
		## calculate idf for every word.
		idf_arr = []
		for i in range(self.term_frequency.shape[1]):
			word_count = (self.term_frequency[:, i] != 0).sum() + 1
			idf = np.log(self.document_count / word_count)
			idf_arr.append(idf)
		return idf_arr

	def analyze_frequencies(self, term_frequency):
		sum_dict = {}
		word_dict = {}
		for i in range(term_frequency.shape[1]):
			column_sum = term_frequency[:, i].sum()
			if column_sum in sum_dict:
				sum_dict[column_sum] = sum_dict[column_sum] + 1
			else:
				sum_dict[column_sum] = 1
				word_dict[column_sum] = self.words_inverted[i]
		for column_sum, count in sum_dict.iteritems():
			try:
				print("Sum: {}, Count: {}, Word: {}".format(column_sum, count, word_dict[column_sum]))
			except Exception:
				traceback.print_exc() 


	def get_tfidf(self):
		return self.tf_idf


	def build_document_word_matrix(self, documents_to_words):
		document_word_matrix = np.zeros((self.document_count, self.word_count))
		for i, document_to_word in enumerate(documents_to_words):
			for entry in document_to_word:
				document_word_matrix[i][entry[0]] = entry[1]
		print("Extracted words")
		print(document_word_matrix.shape)
		return document_word_matrix


	def extract_word_distribution(self):
		extractor = extract.TermExtractor()
		indices = []
		i = 0
		for label in self.raw_labels:
			if label in ["raw.abstract", "title", 'raw.title']:
				indices.append(i)
			i += 1
		if len(indices) > 2:
			indices = indices[1:]
		total = 0
		documents_to_words = []
		for paper_data in self.raw_data:
			paper_text = ''
			for index in indices:
				paper_text += paper_data[index]
				total += len(paper_data[index])
			document_to_words = []
			keywords = extractor(paper_text)
			for keyword in keywords:
				if keyword[2] > 3:
					break
				word_id = self.insert_word(keyword[0])
				word_count = keyword[1]
				self.words_inverted[word_id] = keyword[0]
				document_to_words.append((word_id, word_count))
			documents_to_words.append(document_to_words)
		print("EXtracted total {}".format(total))
		return documents_to_words
		
	def insert_word(self, word):
		if word in self.words:
			return self.words[word]
		self.words[word] = self.word_count
		self.word_count += 1
		return self.word_count - 1

	def get_word_id(self, word):
		if word in self.words:
			return self.words[word]
		return -1

