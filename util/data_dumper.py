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
        else:
            self.folder = folder
        self.dataset = dataset

    def save_matrix(self, matrix, name=None):
        """
        Function that dumps the matrix to a .dat file.

        :param ndarray matrix: Matrix to be dumped.
        :param str matrix_name: Name of the matrix to be dumped.
        """
        matrix = np.array(matrix)
        #print("dumping ")
        if name is None:
            path = self._create_path(self.dataset)
        else:
            path = self._create_path(self.dataset + "-" + name)
        #print(path)
        #matrix.dump(path)
        #print(matrix.shape)
        np.savetxt(path, np.array(matrix, dtype=int), delimiter=",", fmt='%i')
        #print("dumped to %s" % path)

    def load_matrix(self, name=None):
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
        if name is None:
            path = self._create_path(self.dataset)
        else:
            path = self._create_path(self.dataset + "-" + name)
        #print("trying to load %s" % path)
        try:
            matrix = np.loadtxt(path,dtype=int, delimiter=',')
            #print(matrix.shape)
            res = (True, matrix)

        #    print("loaded from %s" % path)
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
