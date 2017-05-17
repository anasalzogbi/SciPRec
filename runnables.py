#!/usr/bin/env python

import sys
import getopt

from util.data_parser import DataParser
from util.keyword_extractor import KeywordExtractor
from lib.peer_extractor import PeerExtractor
#from lib.svr import SVR
from lib.lda_recommender import LDARecommender
from lib.svr_cluster import SVRCluster
# dp = DataParser('citeulike-a')
# labels, data = dp.get_raw_data()
# e = KeywordExtractor(labels, data)
# tf_idf = e.tf_idf
# peer_extractor = PeerExtractor(dp.ratings, tf_idf, 'least_k', 'cosine', 10)
start_id = -1
end_id = -1
argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv,"hs:e:",["start_id=","end_id="])
except getopt.GetoptError:
	print 'runnables.py -s <start_id> -e <end_id>'
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print 'runnables.py -s <start_id> -e <end_id>'
		sys.exit()
	elif opt in ("-s", "--start_id"):
		start_id = arg
	elif opt in ("-e", "--end_id"):
		end_id = arg
print(start_id)
print(end_id)
#svr = SVRCluster(int(start_id),int(end_id))