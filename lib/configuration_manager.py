#!/usr/env/bin python
"""
This module is used to load configurations.
"""
import json
import os


class ConfigurationManager(object):
    """
    A class that will be used to setup the configuration.
    Available configurations:
    - Parameters:
        - number_of_topics: int
        - number_of_peers: int
        - min_similarity_threshold: float
        - max_similarity_threshold: float
        - first_user: int (zero-based index)
        - last_user: int (zero-based index)
        - dataset: string (dummy, citeulike-a, citeulike-t)
    - methods:
        - paper_presentation: string ("lda")
        - sampling: string ("least_similar_k")
        - pair_formulation: string
        - learning: string
    - metrics:
        - ndcg: list of int, give different values for k (ndcg@k)
        - mrr: int, give the values for k (mrr@k)
        - recall: list of int, give different values for k (recall@k)
        - AUC: Boolean (True, or False)
    """

    def __init__(self):
        """
        Constructs a configuration from the config/ directory.
        """
        base_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(os.path.dirname(base_dir), 'config/configuration.json')) as data_file:
            self.config_dict = json.load(data_file)['configuration']

    def get_number_of_topics(self):

        return self.config_dict['parameters']['number_of_topics']

    def get_number_of_peers(self):

        return self.config_dict['parameters']['number_of_peers']

    def get_min_similarity_threshold(self):

        return self.config_dict['parameters']['min_similarity_threshold']

    def get_max_similarity_threshold(self):
        return self.config_dict['parameters']['max_similarity_threshold']

    def get_first_user(self):
        return self.config_dict['parameters']['first_user']

    def get_last_user(self):
        return self.config_dict['parameters']['last_user']

    def get_dataset(self):
        return self.config_dict['parameters']['dataset']

    def get_paper_presentation(self):

        return self.config_dict['methods']['paper_presentation']

    def get_sampling(self):

        return self.config_dict['methods']['sampling']

    def get_ndcg_values(self):

        return self.config_dict['metrics']['ndcg']

    def get_mrr_values(self):

        return self.config_dict['metrics']['mrr']

    def get_recall_values(self):

        return self.config_dict['metrics']['recall']

    def get_AUC(self):

        return self.config_dict['metrics']['AUC']

