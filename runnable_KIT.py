#!/usr/bin/env python



import os
import sys
import socket
import datetime
import argparse
import numpy as np
import multiprocessing
import itertools as it

from memory_profiler import profile

from lib.ltr import svm_ltr_predict
from lib.ltr import svm_ltr_train
from lib.rocchio import rocchio_predict
from lib.rocchio import rocchio_train

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

#@profile
def recommend_user(user):

	experiments_hashset={}
	#print("Calculating for user {}".format(user))
	experiments_results = []
	for (min_sim, max_sim, peers) in parameters:
		experiment_results=[]
		for fold in range(k_folds):
			fold_results = []
			_, user_fold_test_indices, user_training_ratings, user_test_ratings = get_user_fold_indices_and_data(user, fold, train_indices_shared, test_indices_shared, ratings_shared, k_folds)
			#print("***** Model learning")

			trained_model = svm_ltr_train(peer_sampling_method, user_training_ratings, similarity_matrix_shared, experiments_hashset, min_sim, max_sim, peers, pair_build_method, peer_scoring_method, document_matrix_shared)
			#trained_model = rocchio_train(user_training_ratings, document_matrix_shared, experiments_hashset)
			if len(trained_model)==1:
				print "**** Experiment loaded"
				experiment_results.append(experiments_hashset[trained_model[0]])
			else:
				clf, experiment_hash = trained_model
				user_test_documents = document_matrix_shared[user_fold_test_indices]

				#print("***** Predicting")
				predictions = svm_ltr_predict(clf, user_test_documents)
				#predictions = rocchio_predict(clf, user_test_documents)

				ndcg_at_10, mrr_at_10 = calculate_ndcg_mrr(predictions, user_fold_test_indices, 10, user_test_ratings)
				fold_results.append(ndcg_at_10)
				fold_results.append(mrr_at_10)
				recall_xs = [10, 50, 100, 200]
				for recall_x in recall_xs:
					fold_results.append(
						calculate_top_recall(predictions, user_fold_test_indices, recall_x, user_test_ratings))
				experiment_results.append(fold_results)
				experiments_hashset[experiment_hash] = fold_results
			del fold_results
		experiments_results.append( ((min_sim, max_sim, peers), [user]+np.mean(np.array(experiment_results),axis=0 ).tolist() ))
		#del experiment_results
		print "Worker {}- Experiment: User:{}, min_sim:{}, max_sim:{}, peers:{}, result:{}".format(worker_id, user,min_sim, max_sim, peers,np.mean(np.array(experiment_results),axis=0 ).tolist())

	try:
		''' increment the global counter '''
		global counter
		# += operation is not atomic, so we need to get a lock:
		with counter.get_lock():
			counter.value += 1
		print ("Worker {}, Progress: {}/{}".format(worker_id, counter.value, end_id-start_id))
		del experiments_hashset
		return experiments_results
	except NameError:
		del experiments_hashset
		return experiments_results



start_id = 0
end_id = -1

sim_min = [0.05]

sim_max = [0.3]

peers = [1]

paper_presentation = "keywords"
peer_scoring_method = "user_based"
processes = 1
worker_id = socket.gethostname()
k_folds = 5

topics_count = 70

similarity_metric =  'cosine'
peer_sampling_method = "least_similar_k"
pair_build_method = "pairs"
num_cores = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help = "Which dataset to use", choices=['dummy', 'citeulike-a', 'citeulike-t'])
parser.add_argument("--paper_presentation", "-paper_pres", help = "Specifies paper's model", choices=['lda', 'keywords','attributes' ])
parser.add_argument("--topics", "-topics", help = "Specifies the number of latent topics for lda paper's representation", type =int)
parser.add_argument("--peer_sampling", "-peer_sam", help = "Specifies the peer sampling method", choices=['random', 'least_k','least_similar_k' ])
parser.add_argument("--peer_scoring_method", "-peer_scoring", help = "Peer scoring method", choices=['pair_based', 'user_based'])
parser.add_argument("--pair_build_method", "-pair_building", help = "Pair building method", choices=['pairs', 'singles'])

parser.add_argument("--processes", "-cores", help = "The number of processes (cores), -1 for using all available cores, default is 1", choices = [str(i) for i in [-1]+range(1,num_cores+1)])

parser.add_argument('--peers', '-p', nargs='+', type=int,help= "The number of peer papers added for each relevant paper, accepts multiple values")

parser.add_argument('--min_sim', nargs='+', type=float,help= "The minimum threshols for peer sampling, accepts multiple values")

parser.add_argument('--max_sim', nargs='+', type=float,help= "The maximum threshols for peer sampling, accepts multiple values")

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

if args.peers:
	peers = args.peers

if args.min_sim:
	sim_min = args.min_sim

if args.max_sim:
	sim_max = args.max_sim

if args.starting_user:
	start_id = args.starting_user
if args.ending_user:
	end_id = args.ending_user
if args.worker_id:
	worker_id = args.worker_id

if args.paper_presentation:
	paper_presentation = args.paper_presentation
if args.peer_sampling:
	peer_sampling_method = args.peer_sampling
if args.peer_scoring_method:
	peer_scoring_method = args.peer_scoring_method
if args.pair_build_method:
	pair_build_method = args.pair_build_method
if args.topics:
	if args.topics <=0 or args.topics >=500:
		print "Topics count {} must be a positive integer less than 500".format(args.topics)
	else:
		topics_count = args.topics

print("*** Starting user: {}".format(start_id))
print("*** Ending user: {}".format(end_id))
parameters = [ (mins, maxs,p) for (mins, maxs,p) in it.product(sim_min, sim_max, peers) if mins < maxs]
print ("*** Number of experiments per user: {}".format(sum(1 for _ in parameters)))

start_id = int(start_id)
end_id = int(end_id)

parser = DataParser(dataset, topics_count, paper_presentation)
document_matrix_shared = parser.get_document_word_distribution()
print ("**** Document Matrix: {}".format(document_matrix_shared.shape))


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
np.set_printoptions(precision=3)
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

for (param, results) in sorted(d.items(), key=lambda x: x[0]):
	path = create_path('results/{}'.format(dataset), param)
	np.savetxt(path, np.array(results+[[-1]+ np.mean(np.array(results)[:,1:],axis=0).tolist() ]), delimiter=",", fmt='%1.3f')
	print "Experiment: min_sim:{}, max_sim:{}, peers:{}, result:{}".format(param[0],param[1], param[2], ["{0:0.3f}".format(i) for i in np.mean(np.array(results)[:,1:],axis=0).tolist()])




