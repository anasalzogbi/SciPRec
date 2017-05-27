#!/usr/bin/env python



import os
import sys
import socket
import datetime
import argparse
import numpy as np
import multiprocessing
import itertools as it


from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from util.data_parser import DataParser
from util.data_dumper import DataDumper
from util.evaluator import calculate_ndcg_mrr
from util.evaluator import calculate_top_recall
from util.similarity_calculator import calculate_pairwise_similarity

from lib.peer_extractor import get_user_peer_papers
from lib.peer_extractor import build_pair_vector_label_user_sim
from lib.folds_splitter import get_user_fold_indices_and_data



def create_path(folder, param_list):
	"""
	Function creates a string uniquely representing the matrix it also
	uses the config to generate the name.

	:param str matrix_name: Name of the matrix.
	:param int n_rows: Number of rows of the matrix.
	:returns: A string representing the matrix path.
	:rtype: str
	"""
	base_dir = os.path.dirname(os.path.realpath(__file__))
	direct = os.path.join(base_dir, folder)
	if not os.path.exists(direct):
		os.makedirs(direct)
	file_name = "mins_{}_maxs_{}_peers_{}.csv".format(param_list[0], param_list[1], param_list[2])

	return os.path.join(direct, file_name)


def recommend_user(user):
	np.set_printoptions(precision=3)
	experiments_hashset={}
	#print("Calculating for user {}".format(user))
	experiments_results = []
	for (min_sim, max_sim, peers) in parameters:
		experiment_results=[]
		for fold in range(k_folds):
			fold_results = []
			_, user_fold_test_indices, user_training_ratings, user_test_ratings = get_user_fold_indices_and_data(user, fold, train_indices_shared, test_indices_shared, ratings_shared, k_folds)
			# print("***** Building user's pairs")
			relevant_papers, peer_papers, pairs_scores = get_user_peer_papers(peer_extraction_method, user_training_ratings, similarity_matrix_shared, min_sim, max_sim, peers)

			experiment_hash = hash(frozenset(zip(relevant_papers, peer_papers)))
			if experiment_hash not in experiments_hashset:
				pairs = zip(relevant_papers, peer_papers, pairs_scores)
				feature_vectors = []
				labels = []
				i = 0
				for pair in pairs:
					feature_vector, label = build_pair_vector_label_user_sim(pair, document_matrix_shared)
					feature_vectors.append(feature_vector[0])
					feature_vectors.append(feature_vector[1])
					labels.append(label[0])
					labels.append(label[1])
					i += 1
				#print("*** PAIRS BUILT ***")
				feature_vectors = np.array(feature_vectors)
				# Using grid search for fitting hyper barameters (Penalty, C)
				tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [0.01, 1, 10]}]
				grid_clf = GridSearchCV(LinearSVC(dual=False, tol=0.0001, random_state=41), tuned_parameters, cv=3, scoring='recall')
				feature_vectors = preprocessing.scale(feature_vectors)
				grid_clf.fit(feature_vectors, labels)
				# print("Best parameters set found on development set: {}").format(grid_clf.best_estimator_)
				# Using LinnearSVC instead of SVC, it uses the implementation of liblinear, should be faster for linear models.
				#print "*** FITTED SVR FOR USER  ***"
				#print("took {}".format(datetime.datetime.now() - t0))

				user_test_documents = document_matrix_shared[user_fold_test_indices]

				# print("***** Predicting")
				predictions = grid_clf.decision_function(user_test_documents)
				ndcg_at_10, mrr_at_10 = calculate_ndcg_mrr(predictions, user_fold_test_indices, 10, user_test_ratings)
				fold_results.append(ndcg_at_10)
				fold_results.append(mrr_at_10)
				recall_xs = [10, 50, 100, 200]
				for recall_x in recall_xs:
					fold_results.append(
						calculate_top_recall(predictions, user_fold_test_indices, recall_x, user_test_ratings))
				experiment_results.append(fold_results)
				experiments_hashset[experiment_hash] = fold_results
			else:
				print "**Experiment loaded"
				experiment_results.append(experiments_hashset[experiment_hash])
		experiments_results.append( ((min_sim, max_sim, peers), [user]+np.mean(np.array(experiment_results),axis=0 ).tolist() ))
		print "Worker {}- Experiment: User:{}, min_sim:{}, max_sim:{}, peers:{}, result:{}".format(worker_id, user,min_sim, max_sim, peers,np.mean(np.array(experiment_results),axis=0 ).tolist())
				#print("User {}, training {}, test {}, test_pos {} :".format(user, len(feature_vectors), len(test_documents), sum(test_data[user])))
			#print(str(['{:^7}'.format(v) for v in ["NDCG@10", "MRR", "REC@10", "REC@50", "REC@100", "REC@200"]]))
			#print(str(['{:06.5f}'.format(v) for v in [user]+results ]))
	#print results
	try:
		''' increment the global counter '''
		global counter
		# += operation is not atomic, so we need to get a lock:
		with counter.get_lock():
			counter.value += 1
		print ("Worker {}, Progress: {}/{}".format(worker_id, counter.value, end_id-start_id))
		return experiments_results
	except NameError:
		return experiments_results



start_id = 0
end_id = -1

sim_min_st = 0.05
sim_min_end = 0.1
sim_min_step = 0.05

sim_max_st = 0.3
sim_max_end = 1
sim_max_step = 0.3

peers_st = 1
peers_end = 50
peers_step = 10

paper_presentation = "keywords"
processes = 1
worker_id = socket.gethostname()
k_folds = 5

similarity_metric =  'cosine'
peer_extraction_method = "least_similar_k"
num_cores = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help = "Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
parser.add_argument("--processes", "-cores", help = "The number of processes (cores), -1 for using all available cores, default is 1", choices = [str(i) for i in [-1]+range(1,num_cores+1)])
parser.add_argument("--peers_start", "-pe_st", help = "The number of peer papers added for each relevant paper, the start value of the experiments", type = int)
parser.add_argument("--peers_end", "-pe_end", help = "The number of peer papers added for each relevant paper, the end value of the experiments", type = int)
parser.add_argument("--peers_step", "-pe_stp", help = "The peer's step value of the experiments", type = int)
parser.add_argument("--min_sim_st", "-mn_st", help = "The minimum threshold for the peers, the start value of the experiments", type = float)
parser.add_argument("--min_sim_end", "-mn_end", help = "The minimum threshold for the peers, the end value of the experiments", type = float)
parser.add_argument("--min_sim_step", "-mn_stp", help = "The step value of he minimum smilarity threshold of the experiments", type = float)
parser.add_argument("--max_sim_st", "-mx_st", help = "The maximum threshold for the peers, the start value of the experiments", type = float)
parser.add_argument("--max_sim_end", "-mx_end", help = "The maximum threshold for the peers, the end value of the experiments", type = float)
parser.add_argument("--max_sim_step", "-mx_stp", help = "The step value of he maximum smilarity threshold of the experiments", type = float)
parser.add_argument("--worker_id", "-w", help = "The worker id, used to calculate overall progress", type = str)
parser.add_argument("--starting_user", "-s", help="The index of the first user (zero based)", type=int)
parser.add_argument("--ending_user", "-e", help="The index of the last user (zero based)", type=int)

args = parser.parse_args()

if args.processes:
	processes = int(args.processes)
else:
	print "*** Running on a single core"
if args.dataset:
	dataset = args.dataset
if args.peers_start:
	peers_st = args.peers_start
if args.peers_end:
	peers_end = args.peers_end
if peers_end <= peers_st :
	print "Peers_end {} must be greater than Peers_start {}"
	sys.exit()
if args.peers_step:
	peers_step = args.peers_step

if args.min_sim_st:
	sim_min_st = args.min_sim_st
if args.min_sim_end:
	sim_min_end = args.min_sim_end
if args.min_sim_step:
	sim_min_step = args.min_sim_step
if sim_min_end <= sim_min_st :
	print "sim_min_end {} must be greater than sim_min_st {}"
	sys.exit()

if args.max_sim_st:
	sim_max_st = args.max_sim_st
if args.max_sim_end:
	sim_max_end = args.max_sim_end
if args.max_sim_step:
	sim_max_step = args.max_sim_step
if sim_max_end <= sim_max_st :
	print "sim_max_end {} must be greater than sim_max_st {}"
	sys.exit()

if args.starting_user:
	start_id = args.starting_user
if args.ending_user:
	end_id = args.ending_user
if args.worker_id:
	worker_id = args.worker_id

print("*** Starting user: {}".format(start_id))
print("*** Ending user: {}".format(end_id))
parameters = [ (mins, maxs,p) for (mins, maxs,p) in it.product(np.arange(sim_min_st, sim_min_end, sim_min_step), \
		   np.arange(sim_max_st, sim_max_end, sim_max_step), \
		   np.arange(peers_st, peers_end, peers_step)) if mins < maxs]
print ("*** Number of experiments per user: {}".format(sum(1 for _ in parameters)))

start_id = int(start_id)
end_id = int(end_id)

parser = DataParser(dataset, 20, paper_presentation)
document_matrix_shared = parser.get_document_word_distribution()


ratings_shared = parser.get_ratings_matrix()
t0 = datetime.datetime.now()

print("*** Loading data ***")
loader = DataDumper(dataset, 'splits')
_, train_indices_shared = (loader.load_matrix('train_indices'))
_, test_indices_shared = (loader.load_matrix('test_indices'))


print("*** Calculating Similarity ***")
similarity_matrix_shared = calculate_pairwise_similarity(ratings_shared.shape[1], similarity_metric, document_matrix_shared)
print("*** Creating parallel job")
print("*** Running experiments on users: ")

results = []
if processes == 1:
	print("*** Single core ")
	for user in range(start_id, end_id):
		results.append(recommend_user(user))
		print ("**** Progress: {}/{}".format(user, end_id - start_id))

else:
	counter = multiprocessing.Value('i', 0)
	if processes == -1:
		print("*** All cores ({})".format(num_cores))
		p = multiprocessing.Pool()
	else:
		print("*** {} cores ".format(processes))
		p = multiprocessing.Pool(processes)
	results = p.map(recommend_user, range(start_id, end_id))
print("*** Done all users users, took {} ".format(datetime.datetime.now()-t0))
d = {}
for user_results in results:
	for (param,result) in user_results:
		try:
			d[param].append(result)
		except KeyError:
			d[param] = [result]
for (param, results) in d.items():
	path = create_path('results/{}'.format(dataset), param)
	np.savetxt(path, np.array(results), delimiter=",", fmt='%1.3f')