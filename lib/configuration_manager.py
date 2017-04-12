#!/usr/env/bin python
"""
This module is used to load configurations.
"""
import json
import os


class ConfigurationManager(object):
    """
    A class that will be used to setup the configuration.
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

    def get_similarity_threshold(self):

        return self.config_dict['parameters']['threshold']

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