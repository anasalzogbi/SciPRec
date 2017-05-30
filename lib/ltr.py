import numpy as np

from sklearn.svm import LinearSVC
from sklearn import preprocessing




from lib.peer_extractor import get_user_peer_papers
from lib.peer_extractor import build_pair_vector_label_user_sim

def svm_ltr_train(peer_sampling_method, user_training_ratings, similarity_matrix_shared, experiments_hashset, min_sim, max_sim, peers, pair_build_method, peer_scoring_method, document_matrix_shared ):
	# print("***** Building user's pairs")

	relevant_papers, peer_papers, pairs_scores = get_user_peer_papers(peer_sampling_method, user_training_ratings, similarity_matrix_shared, min_sim, max_sim, peers, peer_scoring_method)
	experiment_hash = hash(frozenset(zip(relevant_papers, peer_papers)))
	if experiment_hash not in experiments_hashset:
		pairs = zip(relevant_papers, peer_papers, pairs_scores)
		feature_vectors = []
		labels = []
		i = 0
		for pair in pairs:
			feature_vector, label = build_pair_vector_label_user_sim(pair, document_matrix_shared, pair_build_method)
			feature_vectors.append(feature_vector[0])
			feature_vectors.append(feature_vector[1])
			labels.append(label[0])
			labels.append(label[1])
			i += 1
		# print("*** PAIRS BUILT ***")
		feature_vectors = np.array(feature_vectors)
		# Using grid search for fitting hyper barameters (Penalty, C)
		tuned_parameters = [{'penalty': ['l2'], 'C': [0.1]}]
		#clf = GridSearchCV(LinearSVC(dual=False, tol=0.001, random_state=41), tuned_parameters, cv=3, scoring='roc_auc')
		#clf = SVC()
		clf = LinearSVC(dual=False, tol=0.001, random_state=41, penalty='l2', C=0.1)
		# clf= RandomForestClassifier()
		feature_vectors = preprocessing.scale(feature_vectors)
		clf.fit(feature_vectors, labels)
		#print ("learned model: {}".format(sorted(clf.coef_[0], reverse=True)[0:50]))
		#print ("Length: {}, min: {}, max: {}, mean: {}, std: {}".format(len(clf.coef_[0]), min(clf.coef_[0]), max(clf.coef_[0]), np.mean(clf.coef_[0]), np.std(clf.coef_[0])))
		return [clf,experiment_hash]
	else:
		return [experiment_hash]

def svm_ltr_predict(learned_model, user_test_documents):
	return learned_model.decision_function(user_test_documents)
