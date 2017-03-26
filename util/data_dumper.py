#!/usr/bin/env python
"""
This module saves and imports matrices.
"""
import os
import numpy as np
import traceback

class DataDumper(object):
    """
    A class for importing and saving models.
    """
    DATAFOLDER = 'matrices'

    def __init__(self, dataset, folder=None):
        if folder is None:
            self.folder = self.DATAFOLDER
        self.dataset = dataset

    def save_matrix(self, matrix):
        """
        Function that dumps the matrix to a .dat file.

        :param ndarray matrix: Matrix to be dumped.
        :param str matrix_name: Name of the matrix to be dumped.
        """
        print("dumping ")
        path = self._create_path(self.dataset)
        print(path)
        print(matrix.sum())
        np.save(path, matrix)
        print("dumped to %s" % path)

    def load_matrix(self):
        """
        Function that loads a matrix from a file.

        :param dict config: Config that was used to calculate the matrix.
        :param str matrix_name: Name of the matrix to be loaded.
        :param tuple matrix_shape: A tuple of int containing matrix shape.
        :returns:
            A tuple of boolean (if the matrix is loaded or not)
            And the matrix if loaded, random matrix otherwise.
        :rtype: tuple
        """
        path = self._create_path(self.dataset) + '.npy'
        print("trying to load %s" % path)
        try:
            matrix = np.load(path)
            print(matrix.sum())
            res = (True, matrix)

            print("loaded from %s" % path)
            return res
        except Exception:
            print("File not found, %s will initialize randomly" % path)
            traceback.print_exc()
            return (False, None)

    def _create_path(self, matrix_name):
        """
        Function creates a string uniquely representing the matrix it also
        uses the config to generate the name.

        :param str matrix_name: Name of the matrix.
        :param int n_rows: Number of rows of the matrix.
        :returns: A string representing the matrix path.
        :rtype: str
        """
        path = matrix_name
        base_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(os.path.dirname(base_dir), self.folder, path)
