import numpy as np
from util.top_similar import TopSimilar as TopRecommendations

def calculate_ndcg_mrr(predictions, user_test_indices, k, user_test_data):
	dcg = 0.0
	idcg = 0.0
	mrr = 0.0
	spairs = sorted(zip(predictions, user_test_data[user_test_indices]), key=lambda l: l[0], reverse=True)
	#print list(map(lambda x: x[1] , spairs))[0:10]
	for pos_index, (score, label) in enumerate(spairs):
		dcg += label / np.log2(pos_index + 2)
		idcg += 1 / np.log2(pos_index + 2)
		if label == 1 and mrr == 0.0:
			mrr = 1.0 / (pos_index + 1) * 1.0
		if pos_index + 1 == k:
			break
	if idcg != 0:
		return (dcg / idcg), mrr
	return 0, mrr
	"""
	dcg = 0.0
	idcg = 0.0
	mrr = 0.0
	top_predictions = TopRecommendations(k)
	for prediction, index in zip(predictions, user_test_indices):
		top_predictions.insert(index, prediction)
	recommendation_indices = top_predictions.get_indices()
	for pos_index, index in enumerate(recommendation_indices):
		hit_found = False
		dcg += user_test_data[index] / np.log2(pos_index + 2)
		idcg += 1 / np.log2(pos_index + 2)
		if user_test_data[index] == 1 and mrr == 0.0:
			mrr = 1.0 / (pos_index + 1) * 1.0
		if pos_index + 1 == k:
			break
	if idcg != 0:
		return (dcg / idcg), mrr
	return 0, mrr
	"""

def calculate_top_recall(predictions, user_test_indices, k, user_test_data):
	recall = 0.0
	top_predictions = TopRecommendations(k)
	for prediction, index in zip(predictions, user_test_indices):
		top_predictions.insert(index, prediction)
	nonzeros = user_test_data.nonzero()[0]
	denom = len(nonzeros) * 1.0
	for index in top_predictions.get_indices():
		if index in nonzeros:
			recall += 1.0
	if recall == 0:
		return 0
	return recall / min(denom, k)