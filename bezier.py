# -*- coding: utf-8 -*-
"""This module implements several algorithms to evaluate Bezier curves.

The main functions are:
polyeval_bezier - It directly evaluates the polynomial describing the curve.
bezier_subdivision - It approximates the curve by performing sucesive
   subdivisions in the domain's interval.
backward_differences_bezier - Another approximation method. From the initial
   points of the curve, it computes the rest of the interval.
"""

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


def bernstein_recursive(n, k, t):
    """Helper method. Computes the recurrence relation
    B_i^{n+1}(t) = t * B_{i-1}(t) + (1 - t) * B_i^n(t)

    Parameters
    ----------
    n : integer. Corresponds to the degree of the curve.
    k : integer. Corresponds to the number of monomial in the polynomial.
    t : array of float. The instants in which we evaluate the curve.

    Returns
    -------
    array of float of the same dim as t. The result of the relation.

    """
    if n == 0 and k == 0:
        return 1
    if k == -1 or k == n + 1:
        return 0
    if (n - 1, k - 1) in RECURSIVE_BERNSTEIN_DICT:
        b1 = RECURSIVE_BERNSTEIN_DICT[(n - 1, k - 1)]
    else:
        b1 = bernstein_recursive(n - 1, k - 1, t)
        RECURSIVE_BERNSTEIN_DICT[(n - 1, k - 1)] = b1
    if (n - 1, k) in RECURSIVE_BERNSTEIN_DICT:
        b2 = RECURSIVE_BERNSTEIN_DICT[(n - 1, k)]
    else:
        b2 = bernstein_recursive(n - 1, k, t)
        RECURSIVE_BERNSTEIN_DICT[(n - 1, k)] = b2
    return t * b1 + (1 - t) * b2


def polyeval_bezier(P, num_points, algorithm):
    """Directly evaluates the polynomial describing the curve. The algorithms
    are described in Prautzsch H.,et al. Bézier and B-Spline Techniques.

    Parameters
    ----------
    P : (n+1)xdim array of float. The control points of the curve.
    num_points : integer. The number of points to evaluate.
    algorithm : string. May be:
        direct - simply evaluates \sum_{i=0}^n b_i * B_i^n(t), where
          b_i are the control points and B_i^n are the elements of the
          Bézier basis.
        recursive - uses the relation described in bernstein_recursive.
        horner - evaluates the polynomial using the Horner method.
        deCasteljau - performs the de Casteljau's algorithm.

    Returns
    -------
    Nxdim array of float. The curve points.

    """
    P = np.asarray(P, dtype=np.float64).T
    dim, n = P.shape
    n -= 1
    t = np.linspace(0, 1, num_points)
    if algorithm == 'direct':
        solution = direct_eval(P, t, n, num_points)
    elif algorithm == 'recursive':
        solution = recursive_eval(P, t, n, dim, num_points)
    elif algorithm == 'horner':
        solution = horner_eval(P, t, n, dim)
    elif algorithm == 'deCasteljau':
        solution = casteljau(P, t, num_points, n)
    plt.plot(solution[0,:], solution[1,:])
    plt.hold(True)
    plt.plot(P[0,:], P[1,:], '--om')
    return solution.T


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


def recursive_eval(P, t, n, dim, num_points):
    """Helper method of polyeval_bezier. See its docstring for more
    details.

    Parameters
    ----------
    P : (n+1)xdim array of float. The control points of the curve.
    num_points : integer. The number of points to evaluate.
    t : array of float. The sample time points to evaluate.
    n : integer. The degree of the curve
    dim : integer. The dimension of the curve.

    Returns
    -------
    dimxN array of float. The curve points.

    """
    my_sum = np.zeros([dim, num_points])
    for k in xrange(n + 1):
        my_sum += bernstein_recursive(n, k, t) * P[:, k][:, np.newaxis]
    return my_sum


def horner_eval(P, t, n, dim):
    """Helper method of polyeval_bezier. See its docstring for more
    details.

    Parameters
    ----------
    P : (n+1)xdim array of float. The control points of the curve.
    t : array of float. The sample time points to evaluate.
    n : integer. The degree of the curve
    dim : integer. The dimension of the curve.

    Returns
    -------
    dimxN array of float. The curve points.

    """
    # We can concatenate the end point instead of splitting the interval.
    tt = t[:-1]
    onemt = 1 - tt  # Precompute for better performance
    C = np.asarray([binom(n, k) for k in xrange(n + 1)])
    first = np.asarray([np.polyval(C * P[k, :], tt / (onemt))
                        for k in xrange(dim)]) * (onemt)**n
    last = P[:, 0][:, np.newaxis]
    return np.hstack([first, last])[:, ::-1]


def casteljau(P, t, num_points, n):
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
    T = np.repeat(np.atleast_3d(P), num_points, axis=2)
    for i in xrange(n):
        T[:, 0:(n - i), :] = (1 - t) * T[:, 0:(n - i), :] + \
            t * T[:, 1:(n - i) + 1, :]
    return T[:, 0, :]


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
    
