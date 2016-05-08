# -*- coding: utf-8 -*-
"""This module implements several algorithms to evaluate Bezier curves.

The main functions are:
polyeval_bezier - It directly evaluates the polynomial describing the curve.
bezier_subdivision - It approximates the curve by performing sucesive
   subdivisions in the domain's interval.
backward_differences_bezier - Another approximation method. From the initial
   points of the curve, it computes the rest of the interval.
"""

__author__ = "Ana María Martínez Gómez, Víctor Adolfo Gallego Alcalá"

%matplotlib inline
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

"""These global variables are required by the validator. They are supposed to
store frequently computed values.
"""
RECURSIVE_BERNSTEIN_DICT = dict()


def binom(n, k):
    """Binomial coefficient using dictionary. We have used the factorial
    instead of using a dictionary because it is faster this way, given the
    characteristics of the input provided by the validator.

    Parameters
    ----------
    n : integer. Corresponds to the first element in nCk
    k : integer. Corresponds to the second element in nCk.

    Returns
    -------
    integer. Corresponds to nCk.

    """
    return np.math.factorial(n) / np.math.factorial(k) / \
        np.math.factorial(n - k)


def direct_eval(P, t, n, num_points):
    """Helper method of polyeval_bezier. See its docstring for more
    details.

    Parameters
    ----------
    P : (n+1)xdim array of float. The control points of the curve.
    num_points : integer. The number of points to evaluate.
    t : array of float. The sample time points to evaluate.
    n : integer. The degree of the curve

    Returns
    -------
    dimxN array of float. The curve points.

    """
    C = np.asarray([binom(n, k) for k in range(n + 1)])
    T = np.zeros([n + 1, num_points])
    tt = 1 - t  # Precompute for better performance
    for k in xrange(n + 1):
        T[k, :] = t**k * tt**(n - k)
        bernstein = np.einsum('i,ij->ij', C, T)
    return np.dot(P, bernstein)


def bezier_subdivision(P, k, epsilon, lines=False):
    """Wrapper method of bezier_subdivision_aux. It also plots the
    curve

    Parameters
        ----------
        P : (n+1)xdim array of float. The control points of the curve.
        k : integer. Number of iterations.
        epsilon : float. Tolerance threshold.
        lines : boolean. Iff True, just returns the extreme points.

    Returns
    -------
    Nxdim array of float. The curve points. N is variable.

    """
    P = np.asarray(P, dtype=np.float64).T
    solution = bezier_subdivision_aux(P, k, epsilon, lines)
    plt.plot(solution[:,0], solution[:,1])
    return solution

    
def bezier_subdivision_aux(P, k, epsilon, lines=False):
    """It approximates the curve by performing sucesive subdivisions
    in the domain's interval. See Section 3.5 from Prautzsch H.,et al.
    Bézier and B-Spline Techniques for more details.

    Parameters
        ----------
        P : dimx(n+1) array of float. The control points of the curve.
        k : integer. Number of iterations.
        epsilon : float. Tolerance threshold.
        lines : boolean. Iff True, just returns the extreme points.

    Returns
    -------
    Nxdim array of float. The curve points. N is variable.

    """
    dim, n = P.shape
    n -= 1
    if n == 1:
        return P.T

    delta2_b = np.diff(P, n=2, axis=1)
    norm_delta2 = np.max(np.linalg.norm(delta2_b, axis=0))

    if lines and n * (n - 1) / 8 * norm_delta2 < epsilon:
        return np.array([P[0], P[-1]])

    if k == 0 or norm_delta2 < epsilon:
        return P.T
    b_first = np.zeros([dim, n + 1])
    X = np.copy(P)
    T = np.atleast_2d(X)
    b_first[:, 0] = P[:, 0]
    for i in xrange(n):
        T[:, 0:(n - i)] = 0.5 * (T[:, 0:(n - i)] + T[:, 1:(n - i) + 1])
        b_first[:, i + 1] = T[:, 0]
    z1 = bezier_subdivision(b_first.T, k - 1, epsilon, lines)
    z2 = bezier_subdivision(T[:, :].T, k - 1, epsilon, lines)
    return np.vstack([z1[:-1, :], z2])


def backward_differences_bezier(P, m, h=None):
    """Another approximation method. From the initial points of the curve, it
    computes the rest of the interval. The points to evaluate are of the form
    h*k for k=0,...,m.
    See Section 3.6 from Prautzsch H.,et al. Bézier and B-Spline Techniques for
    more details.

    Parameters
        ----------
        P : (n+1)xdim array of float. The control points of the curve.
        m : integer.
        h : float. Step size. If None, h=1/m.

    Returns
    -------
    (m+1)xdim array of float. The curve points.

    """
    if h is None:
        h = 1 / m

    n, dim = P.shape
    n -= 1
    r = m - n
    t0 = np.arange(0, (n + 1) * h, h)

    p_init = direct_eval(P.T, t0, n, t0.shape[0])

    deltas_p = {k: np.diff(p_init, k).T for k in range(n + 1)}

    extended_deltas = dict()
    extended_deltas[n] = np.repeat(deltas_p[n], r, axis=0)

    for k in xrange(1, n + 1):
        indep_terms = extended_deltas[n - k + 1]
        indep_terms[0] += deltas_p[n - k][k]
        extended_deltas[n - k] = np.cumsum(indep_terms, axis=0)

    solution = np.vstack((p_init.T, extended_deltas[0]))
    plt.plot(solution[:,0], solution[:,1])
    plt.hold(True)
    plt.plot(P[0,:], P[1,:], '--om')
    return solution
