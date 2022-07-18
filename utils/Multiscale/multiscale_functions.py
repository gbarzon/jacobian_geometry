# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:39:12 2020

This python file contains functions to evaluate the multiscale closeness
centrality of the nodes in a network. [see  https://doi.org/10.1101/2020.10.02.323030 # noqa
for more information.

@author: Vincent Bazinet
"""

import bct
import numpy as np
import tqdm
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from scipy.linalg import expm, fractional_matrix_power


def transition_probabilities(conn, time_points,
                             method='rwL', verbose=True):
    '''
    Function to compute the transition probabilities of random walkers
    initiated on individual nodes in the network

    Parameters
    ----------
    conn : (n, n) ndarray
        Adjacency matrix of our connectome (structural), where n is the
        number of nodes in the network.
    time_points: (k,) ndarray
        Numpy array listing the time points (If Laplacian method) or damping
        factors (if PageRank method) for which we want a node's transition
        probabilities to be evaluated.

    Returns
    -------
    pr : (k, m, n) ndarray
        Transition probabilities of random walkers initiated on individual
        nodes in the network, where 'k' is the number of time points at which
        the transition probabilities are evaluated, 'm' is the number of nodes
        on which the random walks are initiated.

    '''

    k = len(time_points)
    n = len(conn)

    pr = np.zeros((k, n, n))

    if method == 'pagerank':
        for i in tqdm.trange(k, desc='probabilities') if verbose else range(k):
            for j in range(n):
                pr[i, j, :] = perso_PR(conn, j, time_points[i])

    elif method == 'rwL' or method == 'nL' or method == 'L':
        L = laplacian_matrix(conn, method=method)
        for i in tqdm.trange(k, desc='probabilities') if verbose else range(k):
            pr[i, :, :] = expm(-1 * time_points[i] * L)

    else:
        raise ValueError(("Valid options for the method parameter are: "
                          "\'pagerank\', \'rwL\', \'nL\' or \'L\'."))

    return pr


def perso_PR(A, i, damp, max_it=1000):
    """
    Function that gives you the personalized PageRank vector of a
    specified seed node 'i' and for a specified damping factor 'damp'.

    Parameters
    ----------
    A : ndarray (n,n)
        Adjacency matrix representation of the network of interest, where
        n is the number of nodes in the network.
    i : int
        Index of the seed node.
    damp : float
        Damping factor (one minus the probability of restart). Must be between
        0 and 1.
    max_it: int
        Maximum number of iterations to perform in case the algorithm
        does not converge (if reached, then prints a warning).

    Returns
    -------
    pr : ndarray (n,)
        The personalized PageRank vector of node i, for the selected damping
        factor
    """

    degree = np.sum(A, axis=1)  # out-degrees
    n = len(A)

    # Compute the Transition matrix (transposed) of the network
    W = A/degree[:, np.newaxis]
    W = W.T

    # Initialize parameters...
    pr = np.zeros(n)  # PageRank vector
    pr[i] = 1

    delta = 1
    it = 1
    pr_old = pr.copy()

    # Start Power Iteration...
    while delta > 1e-9 or it < max_it:

        pr = damp*W.dot(pr)
        pr[i] += (1-damp)

        delta = np.sum((pr-pr_old)**2)
        pr_old = pr.copy()
        it += 1

    if it >= max_it:
        print("Maximum number of iterations exceeded")

    return pr


def multiscale_closeness(conn, pr):
    '''
    Function to compute the multiscale closeness centrality of
    individual nodes in a network

    Parameters
    ----------
    conn : (n, n) ndarray
        Adjacency matrix of our connectome (structural), where n is the
        number of nodes in the network.
    pr : (k, m, n) ndarray
        Transition probabilities of random walkers initiated on individual
        nodes in the network, where 'k' is the number of time points at which
        the transition probabilities are evaluated, 'm' is the number of nodes
        on which the random walks are initiated.

    Returns
    -------
    multiscale_closeness : (k, n) ndarray
        Multiscale closeness centralities for each of the 'n' nodes in the
        network and for each one of the 'k' time points.
    sp : (n, n) ndarray
        Shortest paths between pairs of nodes in the network.
    '''

    k = pr.shape[0]
    n = pr.shape[1]

    # Compute the shortest path between every pair of nodes. The topological
    # distance is computed as inverse of the connection weight between the two
    # nodes.
    inv_conn = conn.copy()
    inv_conn[inv_conn > 0] = 1 / inv_conn[inv_conn > 0]
    sp = bct.distance_wei(inv_conn)[0]

    # Compute the multiscale shortest path
    multiscale_sp = np.zeros((k, n))
    for i in range(k):
        for j in range(n):
            multiscale_sp[i, j] = np.average(sp[j, :], weights=pr[i, j, :])

    multiscale_closeness = zscore(1/multiscale_sp, axis=1)

    return multiscale_closeness, sp


def neighborhood_similarity(conn, pr, metric="cosine"):
    '''
    Function to compute the multiscale closeness centrality of
    individual nodes in a network

    Parameters
    ----------
    conn : (n, n) ndarray
        Adjacency matrix of our connectome (structural), where n is the
        number of nodes in the network.
    pr : (k, m, n) ndarray
        Transition probabilities of random walkers initiated on individual
        nodes in the network, where 'k' is the number of time points at which
        the transition probabilities are evaluated, 'm' is the number of nodes
        on which the random walks are initiated.
    metric : str
        Metric to be used to compute the neighborhood similarities between
        pairs of nodes. Cosine is currently the only choice available.
        Default: "cosine".
    Returns
    -------
    nei_similarity : (k, n, n) ndarray
        Neighborhood similarities between every pairs of nodes, for the 'n'
        nodes in the network, and for the 'k' time points.

    '''

    k = pr.shape[0]
    n = pr.shape[1]

    nei_similarity = np.zeros((k, n, n))

    for i in tqdm.trange(k, desc='neighborhood similarity'):
        measure = pr[i, :, :].copy()
        nei_similarity[i, :, :] = 1-cdist(measure, measure, metric=metric)

    return nei_similarity


def laplacian_matrix(A, method='rwL'):
    """
    Function to compute the laplacian matrix of the network.

    Parameters
    ----------
    A : (n, n) ndarray
        Adjacency matrix of the network
    method : str
        Version of the Laplacian matrix to be computed. Available options are
        'rwL', 'nL' or 'L'.

    Returns
    -------
    L : (n, n) ndarray
        Laplacian matrix
    """

    D = np.diag(np.sum(A, axis=0))
    L = D-A

    if method == 'rwL':
        L = np.matmul(np.linalg.matrix_power(D, -1), L)

    elif method == 'nL':
        D_minus_half = fractional_matrix_power(D, -0.5)
        L = np.matmul(D_minus_half, L)
        L = np.matmul(L, D_minus_half)

    elif method != 'L':
        raise ValueError(("Valid options for the method parameter are: "
                          "\'rwL\', \'nL\' or \'L\'."))

    return L
