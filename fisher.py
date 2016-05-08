# -*- coding: utf-8 -*-
"""This module contains the class Fisher to use Fisher's Linear Discriminant"""

__author__ = "Ana María Martínez Gómez, Víctor Adolfo Gallego Alcalá"

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
        
    def c_comp(self, p0, p1, sigma0, sigma1, m0_proy, m1_proy):
        """ Computes the c threshold to assign a class to a (projected) point.
            We obtain the exact roots that are local optima of the error
            estimation and then we choose the one that minimizes it.

        Parameters
        ----------
        p0 : float
            prior probability for the first class
        p1 : float
            prior probability for the second class
        sigma0 : float
            maximum likelihood estimator for the std deviation of the
            projected first class
        sigma1 : float
            maximum likelihood estimator for the std deviation of the
            projected second class
        m0_proy : float
            maximum likelihood estimator for the mean of the projected
            first class
        m1_proy : float
            maximum likelihood estimator for the mean of the projected
            second class

        """
        a = -1 / 2 / sigma0**2 + 1 / 2 / sigma1**2
        b = m0_proy / sigma0**2 - m1_proy / sigma1**2
        c = -1 / 2 * m0_proy**2 / sigma0**2 + 1 / 2 * m1_proy**2 / \
            sigma1**2 + np.log(p0 / sigma0) - np.log(p1 / sigma1)
        pol_der = np.array([2 * a, b])
        disc = np.sqrt(b**2 - 4 * a * c)
        rootp = (-b + disc) / (2 * a)
        rootm = (-b - disc) / (2 * a)
        roots = [rootp, rootm]
        index = np.argmin(np.polyval(pol_der, roots))
        self.c = roots[index]

        # c = average of the proyected means (alternative method)
        # self.c = 0.5 * (m0_proy + m1_proy)

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
        
    def classify_fisher(self, x):
        """ Classifies the given points thanks to the previous training.

        Parameters
        ----------
        x : list of points from the first class, given as
            [[x0,...,xD],...,[z0,...,zD]] where D is the number of dimensions

        Returns
        -------
        list of integers
            [c0,c1,...,cM] : where ci is the class of the ith point for each i
            from 0 to M, and ci = 0 or ci = 1

        """
	x = np.asarray(x)
	return ((x.dot(self.w) >= self.c).astype(int)).tolist()
		

