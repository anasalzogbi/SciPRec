import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def calculate_pairwise_similarity(docs_count, similarity_metric, documents):
		similarity_matrix = np.eye(docs_count)
		if similarity_metric == 'dot':
			# Compute pairwise dot product
			for i in range(docs_count):
				for j in range(i, docs_count):
					similarity_matrix[i][j] = documents[i].dot(documents[j].T)
					similarity_matrix[j][i] = similarity_matrix[i][j]

		if similarity_metric == 'cosine':
			similarity_matrix = cosine_similarity(sparse.csr_matrix(documents), dense_output=True)
		return similarity_matrix

