# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class Fisher:

    """ This class offers methods to train and classify data using the Fisher's
        Linear Discriminant. Based on Chapter 4.1 from C.M. Bishop, Pattern
        Recognition and Machine Learning, Springer, 2007.

    Attributes
    ----------
    w : array of float
        Weight vector. Used to project the data to classify
    c : float
        Threshold used to compare the projections

    """
    def __init__(self):
        """ Attribute intitialization """
        self.w = None
        self.c = None

    def train_fisher(self, x0, x1):
        """ Trains with the sample data provided for the Fisher's Linear
        discriminant. It computes the weight vector w and the threshold c.

        Parameters
        ----------
        x0 : list of points from the first class, given as
            [[x0,...,xD],...,[z0,...,zD]] where D is the number of dimensions
        x1 : list of points from the second class, given as
            [[x0,...,xD],...,[z0,...,zD]] where D is the number of dimensions

        """
        x0, x1 = np.asarray(x0).T, np.asarray(x1).T
        n0, n1 = x0[1].size, x1[1].size
        m0, m1 = np.mean(x0, axis=1), np.mean(x1, axis=1)
        sw = np.cov(x0, ddof=n0 - 1) + np.cov(x1, ddof=n1 - 1)
        w_no_norm = np.linalg.solve(sw, m1 - m0)
        self.w = w_no_norm / np.linalg.norm(w_no_norm)

        # c threshold calc
        n = n0 + n1
        p0, p1 = n0 / n, n1 / n
        m0_proy, m1_proy = self.w.T.dot(m0), self.w.T.dot(m1)
        x0_proy, x1_proy = self.w.T.dot(x0), self.w.T.dot(x1)
        sigma0, sigma1 = np.std(x0_proy), np.std(x1_proy)
        self.c_comp(p0, p1, sigma0, sigma1, m0_proy, m1_proy)

