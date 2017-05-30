import numpy as np


from scipy import spatial

def rocchio_train(user_training_ratings, document_matrix_shared, experiments_hashset):
	positive_papers = user_training_ratings.nonzero()[0]
	experiment_hash = hash(frozenset(positive_papers))
	if experiment_hash not in experiments_hashset:
		return (np.sum(document_matrix_shared[positive_papers], axis =0), experiment_hash)
	else:
		return experiment_hash



def rocchio_predict(learned_model, user_test_documents):
	result = []
	for i in range(len(user_test_documents)):
		result.append(spatial.distance.cosine(learned_model, user_test_documents[i]))
	return result
