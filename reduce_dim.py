# -*- coding: utf-8 -*-
"""This module implements two mecanisms for dimensionality reduction.

The first one is LDA, which performs the Linear Discrimant Analysis, and the
second is PCA, which implements the Principal Component Analysis.

"""

from __future__ import division
import numpy as np
from scipy.linalg import eigh


class LDA:

    """This class performs the Linear Discrimant Analysis.

    Based on Section 4.1.6 from C.M. Bishop, Pattern Recognition and Machine
    Learning, Springer, 2007.

    Attributes
    ----------
    m_LDA : array of floats with size D (the number of columns of X_train).
        Mean of the training data samples.
    w_LDA : array of floats with dimension D x reduced_dim.
        Matrix of weight vectors.

    Methods
    -------
    fit : trains the model.
    transform : applies the projection matrix to the given points.

    Examples
    --------
    import reduce_dim as rd

    X_train = np.asarray([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])

    y_train = np.asarray([1, 1, 0, 0, 2])

    ins = rd.LDA()

    ins.fit(X_train, y_train, 1)

    ins.w_LDA
        array([[ 0.64883266],
               [ 0.18472308]])

    X = np.asarray([[5, -3], [2, 8], [9, 6], [-1, -3]])

    ins.transform(X)
        array([[ 0.07849297],
               [ 0.16394891],
               [ 4.33633136],
               [-3.81450299]])

    """

    def __init__(self):
        """Initializes to None all of its attributes, as they are computed in
        the fit method.

        """
        self.m_LDA = None
        self.W_LDA = None

    def fit(self, X_train, y_train, reduced_dim):
        """Trains the model, i.e., it computes the weight matrix w_LDA.

        Parameters
        ----------
        X_train : array of floats of dimension N x D, where N is the number
            of data points and D the number of features.
            The training data points.
        y_train : array of ints of dimensions N.
            The classes of each data point.
        reduced_dim : float.
            The dimension of the subspace in which we want to project.

        """
        dim = np.size(X_train[0, :])
        K = y_train.max() + 1
        self.m_LDA = np.mean(X_train, axis=0)
        sums = np.zeros([dim, K])
        # We compute the class means iterating over dimension using this trick:
        for k in range(dim):
            sums[k, :] = np.bincount(y_train, X_train[:, k])
        N_class = np.bincount(y_train)
        class_means = sums/N_class

        S_B, S_W = np.zeros([dim, dim]), np.zeros([dim, dim])
        for k in range(K):
            dif = (self.m_LDA - class_means[:, k])[:, np.newaxis]
            S_B += N_class[k] * np.dot(dif, dif.T)
        X_train -= self.m_LDA
        # We compute S_T instead of S_W to improve performance
        S_T = np.dot(X_train.T, X_train)
        S_W = S_T - S_B

        _, eigenvectors = eigh(S_B, S_W, eigvals=(dim - reduced_dim, dim - 1),
                               overwrite_a=True, overwrite_b=True)
        self.W_LDA = eigenvectors[:, ::-1]

    def transform(self, X):
        """Applies the projection matrix w_LDA to the given points.

        Parameters
        ----------
        X : array of floats of dimension M x D, where M is the number of points
        and D the number of features.
            The data points.

        Returns
        -------
        array of float of dimension M x reduced_dim.
            The projected data points.

        """
        return np.dot(X - self.m_LDA, self.W_LDA)


class PCA:

    """This class performs the Principal Component Analysis.

    Based on Section 12.1 from C.M. Bishop, Pattern Recognition and Machine
    Learning, Springer, 2007.

    Attributes
    ----------
    m_PCA : array of floats with size D (the number of columns of X_train).
        Mean of the training data samples.
    w_PCA : array of floats with dimension D x reduced_dim.
        Matrix of weight vectors.

    Methods
    -------
    fit : trains the model.
    transform : applies the projection matrix to the given points.

    Examples
    --------
    import reduce_dim as rd

    X_train = np.asarray([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])

    y_train = np.asarray([1, 1, 0, 0, 2])

    ins = rd.PCA()

    ins.fit(X_train, 1)

    ins.w_PCA
        array([[-0.2036295,  0.97904802]])

    X = np.asarray([[5, -3], [2, 8], [9, 6], [-1, -3]])

    ins.transform(X)
        array([[-6.86897593],
               [ 4.51144079],
               [ 1.12793827],
               [-5.64719895]])

    """

    def __init__(self):
        """Initializes to None all of its attributes, as they are computed in
        the fit method.

        """
        self.m_PCA = None
        self.W_PCA = None

    def fit(self, X_train, reduced_dim):
        """Trains the model, i.e., it computes the weight matrix w_PCA.

        Parameters
        ----------
        X_train : array of floats of dimension N x D, where N is the number
            of data points and D the number of features.
            The training data points.
        reduced_dim : float.
            The dimension of the subspace in which we want to project.

        """
        dim = np.size(X_train[0, :])
        self.m_PCA = np.mean(X_train, axis=0)
        X_train -= self.m_PCA
        S = np.dot(X_train.T, X_train)
        _, eigenvectors = eigh(S, eigvals=(dim - reduced_dim, dim - 1),
                               overwrite_a=True)
        self.W_PCA = eigenvectors[:, ::-1]

    def transform(self, X):
        """Applies the projection matrix w_PCA to the given points. It projects
        to the reduced_dim-subspace which maximizes the varianze between the
        projected points.

        Parameters
        ----------
        X : array of floats of dimension M x D, where M is the number of points
        and D the number of features.
            The data points.

        Returns
        -------
        array of float of dimension M x reduced_dim.
            The projected data points.

        """
        return np.dot(X - self.m_PCA, self.W_PCA)
