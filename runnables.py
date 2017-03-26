#!/usr/bin/env python

import sys
from util.data_parser import DataParser
from util.keyword_extractor import KeywordExtractor
from lib.peer_extractor import PeerExtractor
from lib.svr import SVR
# dp = DataParser('citeulike-a')
# labels, data = dp.get_raw_data()
# e = KeywordExtractor(labels, data)
# tf_idf = e.tf_idf
# peer_extractor = PeerExtractor(dp.ratings, tf_idf, 'least_k', 'cosine', 10)


svr = SVR()