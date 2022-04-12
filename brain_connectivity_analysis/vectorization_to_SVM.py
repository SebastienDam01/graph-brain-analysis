#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:51:23 2022

@author: sdam
"""

import pickle
import random
import copy
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import os

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

os.chdir('..')


from utils.utils import printProgressBar

# Load variables from data_preprocessed.pickle
with open('manage_data/data_preprocessed.pickle', 'rb') as f:
    connectivity_matrices, controls, patients, controls_count, patients_count, subject_count = pickle.load(f)

# Load volumes from volumes_preprocessed.picke
with open('manage_data/volumes_preprocessed.pickle', 'rb') as f:
    volumes_ROI = pickle.load(f)

nb_ROI = len(connectivity_matrices[patients[0]])

# TEMPORARY
subjects_to_delete = ['lgp_081LJ', 
                      'lgp_164AS',
                      'lgp_079LG', 
                      'lgp_073RN', 
                      'lgp_075TH', 
                      'lgp_071LA', 
                      'lgp_072DF', 
                      'lgp_076FO']

for subject in subjects_to_delete:
    if subject in patients:
        patients.remove(subject)
    else:
        controls.remove(subject)
        
subject_count = subject_count - len(subjects_to_delete)
patients_count = patients_count - len(subjects_to_delete)

connectivity_matrices = dict([(key, val) for key, val in 
           connectivity_matrices.items() if key not in subjects_to_delete])

# Density fibers connectivity matrices
def nb_fiber2density(matrices, volumes):
    n = list(matrices.items())[0][1].shape[0]
    densities = copy.deepcopy(matrices)
    densities = dict((k,v.astype(float)) for k,v in densities.items())
    for subject, mat in densities.items():
        for i in range(n):
            for j in range(n):
                mat[i, j] = mat[i, j] / (volumes[subject][i, 0] + volumes[subject][j, 0])
    
    return densities

connectivity_matrices = nb_fiber2density(connectivity_matrices, volumes_ROI)

def get_network(matrix, threshold = 0):
    """ 
        Return the network (as a networkx data structure) defined by matrix.
        It is possible to specify a threshold that will disregard all the 
        edges below this threshold when creating the network
    """
    G = nx.Graph()
    N = matrix.shape[0]
    G.add_nodes_from(list(range(N)))
    G.add_weighted_edges_from([(i,j,1.0*matrix[i][j]) for i in range(0,N) for j in range(0,i) \
                                                                   if matrix[i][j] >= threshold])
    return G


# Network measures 
# https://github.com/aestrivex/bctpy
def cuberoot(x):
    '''
    Correctly handle the cube root for negative weights.
    '''
    return np.sign(x) * np.abs(x)**(1 / 3)

def clustering_coefficient(matrix):
    K = np.array(np.sum(np.logical_not(matrix == 0), axis=1), dtype=float)
    ws = cuberoot(matrix)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, set C=0
    C = cyc3 / (K * (K - 1))
    return C

def invert(W, copy=True):
    '''
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        inverted connectivity matrix
    '''
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return W

def efficiency(matrix, local=False):
    def distance_inv_wei(matrix):
        n = len(matrix)
        D = np.zeros((n, n))  # distance matrix
        D[np.logical_not(np.eye(n))] = np.inf

        for u in range(n):
            # distance permanence (true is temporary)
            S = np.ones((n,), dtype=bool)
            G1 = matrix.copy()
            V = [u]
            while True:
                S[V] = 0  # distance u->V is now permanent
                G1[:, V] = 0  # no in-edges as already shortest
                for v in V:
                    W, = np.where(G1[v, :])  # neighbors of smallest nodes
                    td = np.array(
                        [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                    D[u, W] = np.min(td, axis=0)

                if D[u, S].size == 0:  # all nodes reached
                    break
                minD = np.min(D[u, S])
                if np.isinf(minD):  # some nodes cannot be reached
                    break
                V, = np.where(D[u, :] == minD)

        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D
    
    n = len(matrix)
    Gl = invert(matrix, copy=True)  # connection length matrix
    A = np.array((matrix != 0), dtype=int)
    if local:
        E = np.zeros((n,))  # local efficiency
        for u in range(n):
            # V,=np.where(matrix[u,:])		#neighbors
            # k=len(V)					#degree
            # if k>=2:					#degree must be at least 2
            #	e=(distance_inv_wei(Gl[V].T[V])*np.outer(matrix[V,u],matrix[u,V]))**1/3
            #	E[u]=np.sum(e)/(k*k-k)
    
            # find pairs of neighbors
            V, = np.where(np.logical_or(matrix[u, :], matrix[:, u].T))
            # symmetrized vector of weights
            sw = cuberoot(matrix[u, V]) + cuberoot(matrix[V, u].T)
            # inverse distance matrix
            e = distance_inv_wei(Gl[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = cuberoot(e) + cuberoot(e.T)
    
            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency
    
    else:
        e = distance_inv_wei(Gl)
        E = np.sum(e) / (n * n - n)
    return E

def charpath(D, include_diagonal=False, include_infinite=True):
    '''
    The characteristic path length is the average shortest path length in
    the network. The global efficiency is the average inverse shortest path
    length in the network.

    Parameters
    ----------
    D : NxN np.ndarray
        distance matrix
    include_diagonal : bool
        If True, include the weights on the diagonal. Default value is False.
    include_infinite : bool
        If True, include infinite distances in calculation

    Returns
    -------
    lambda : float
        characteristic path length

    Notes
    -----
    The input distance matrix may be obtained with the method `adjacency_matrix`
    by networkx. 
    Characteristic path length is calculated as the global mean of
    the distance matrix D, excludings any 'Infs' but including distances on
    the main diagonal.
    
    Example: 
    >>> adj_matrix_sparse = nx.adjacency_matrix(G_nx) # G_nx is a graph with a 
    networkx data structure 
    >>> distance_matrix = adj_matrix_sparse.todense()
    >>> L = charpath(distance_matrix)
    '''
    D = D.copy()

    if not include_diagonal:
        np.fill_diagonal(D, np.nan)

    if not include_infinite:
        D[np.isinf(D)] = np.nan

    Dv = D[np.logical_not(np.isnan(D))].ravel()

    # mean of finite entries of D[G]
    lambda_ = np.mean(Dv)

    return lambda_

def binarize(G):
    W = G.copy()
    W[W != 0] = 1
    return W

def degree(matrix):
    if not np.array_equal(np.unique(matrix), np.array([0, 1])):
        B = binarize(matrix)
    return B @ np.ones((matrix.shape[0],))

def betweenness_centrality(matrix):
    '''
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    L : NxN np.ndarray
        directed/undirected weighted connection matrix

    Returns
    -------
    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
       The input matrix must be a connection-length matrix, typically
        obtained via a mapping from weight to length. For instance, in a
        weighted correlation network higher correlations are more naturally
        interpreted as shorter distances and the input matrix should
        consequently be some inverse of the connectivity matrix.
       Betweenness centrality may be normalised to the range [0,1] as
        BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n = len(matrix)
    BC = np.zeros((n,))  # vertex betweenness

    for u in range(n):
        D = np.tile(np.inf, (n,))
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        G1 = matrix.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(G1[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also predecessor

            if D[S].size == 0:
                break  # all nodes were reached
            if np.isinf(np.min(D[S])):  # some nodes cannot be reached
                Q[:q + 1], = np.where(np.isinf(D))  # these are first in line
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DP[v] += (1 + DP[w]) * NP[v] / NP[w]

    return BC

def get_rng(seed=None):
    """
    By default, or if `seed` is np.random, return the global RandomState
    instance used by np.random.
    If `seed` is a RandomState instance, return it unchanged.
    Otherwise, use the passed (hashable) argument to seed a new instance
    of RandomState and return it.

    Parameters
    ----------
    seed : hashable or np.random.RandomState or np.random, optional

    Returns
    -------
    np.random.RandomState
    """
    if seed is None or seed == np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, np.random.RandomState):
        return seed
    try:
        rstate =  np.random.RandomState(seed)
    except ValueError:
        rstate = np.random.RandomState(random.Random(seed).randint(0, 2**32-1))
    return rstate

class BCTParamError(RuntimeError):
    pass

def modularity_louvain_und(W, gamma=1, hierarchy=False, seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.

    The Louvain algorithm is a fast and accurate community detection
    algorithm (as of writing). The algorithm may also be used to detect
    hierarchical community structure.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    hierarchy : bool
        Enables hierarchical output. Defalut value=False
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.

    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector. If hierarchical output enabled,
        it is an NxH np.ndarray instead with multiple iterations
    Q : float
        optimized modularity metric. If hierarchical output enabled, becomes
        an Hx1 array of floats instead.

    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    rng = get_rng(seed)

    n = len(W)  # number of nodes
    s = np.sum(W)  # weight of edges
    h = 0  # hierarchy index
    ci = []
    ci.append(np.arange(n) + 1)  # hierarchical module assignments
    q = []
    q.append(-1)  # hierarchical modularity values
    n0 = n

    #knm = np.zeros((n,n))
    # for j in np.xrange(n0+1):
    #    knm[:,j] = np.sum(w[;,

    while True:
        if h > 300:
            raise BCTParamError('Modularity Infinite Loop Style B.  Please '
                                'contact the developer with this error.')
        k = np.sum(W, axis=0)  # node degree
        Km = k.copy()  # module degree
        Knm = W.copy()  # node-to-module degree

        m = np.arange(n) + 1  # initial module assignments

        flag = True  # flag for within-hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Modularity Infinite Loop Style C.  Please '
                                    'contact the developer with this error.')
            flag = False

            # loop over nodes in random order
            for i in rng.permutation(n):
                ma = m[i] - 1
                # algorithm condition
                dQ = ((Knm[i, :] - Knm[i, ma] + W[i, i]) -
                      gamma * k[i] * (Km - Km[ma] + k[i]) / s)
                dQ[ma] = 0

                max_dq = np.max(dQ)  # find maximal modularity increase
                if max_dq > 1e-10:  # if maximal increase positive
                    j = np.argmax(dQ)  # take only one value
                    # print max_dq,j,dQ[j]

                    Knm[:, j] += W[:, i]  # change node-to-module degrees
                    Knm[:, ma] -= W[:, i]

                    Km[j] += k[i]  # change module degrees
                    Km[ma] -= k[i]

                    m[i] = j + 1  # reassign module
                    flag = True

        _, m = np.unique(m, return_inverse=True)  # new module assignments
        # print m,h
        m += 1
        h += 1
        ci.append(np.zeros((n0,)))
        # for i,mi in enumerate(m):	#loop through initial module assignments
        for i in range(n):
            # print i, m[i], n0, h, len(m), n
            # ci[h][np.where(ci[h-1]==i+1)]=mi	#assign new modules
            ci[h][np.where(ci[h - 1] == i + 1)] = m[i]

        n = np.max(m)  # new number of modules
        W1 = np.zeros((n, n))  # new weighted matrix
        for i in range(n):
            for j in range(i, n):
                # pool weights of nodes in same module
                wp = np.sum(W[np.ix_(m == i + 1, m == j + 1)])
                W1[i, j] = wp
                W1[j, i] = wp
        W = W1

        q.append(0)
        # compute modularity
        q[h] = np.trace(W) / s - gamma * np.sum(np.dot(W / s, W / s))
        if q[h] - q[h - 1] < 1e-10:  # if modularity does not increase
            break

    ci = np.array(ci, dtype=int)
    if hierarchy:
        ci = ci[1:-1]
        q = q[1:-1]
        return ci, q
    else:
        return ci[h - 1], q[h - 1]

def participation_coef(W, ci, degree='undirected'):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree

    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P

def average_neighbor_degree(matrix):
    '''
    The average neighbor degree for weighted graphs is defined as 
    k^w_{nn, i} = \frac{1}{s_i}\sum{j \in N(i)}{w_{ij}k_j}
    with s_i the strength of node i and k_j is the degree of node j
    which belongs to N(i).

    Parameters
    ----------
    matrix : NxN np.ndarray
            undirected weighted connection matrix

    Returns
    -------
    k : Nx1 np.darray
        average neighbor degree
        
    Notes
    -----
    
    strength in the formula comes from [2] in http://www.cpt.univ-mrs.fr/~barrat/pnas101.pdf
    '''
    if not np.array_equal(np.unique(matrix), np.array([0, 1])):
        B = binarize(matrix)
    
    n = len(matrix)
    k = np.zeros((n,))
    s = np.sum(matrix, axis=0) # strength
    for i in range(n):
        N, = np.where(matrix[i, :]) # neighbors of i
        w_k = 0
        for j in N:
            w_k += matrix[i, j] * B[j, :] @ np.ones((n,))
        k[i] = (1 / s[i]) * w_k
    return k

def node_curvature(matrix):
    n = len(matrix)
    curvature = np.zeros((n))
    G_nx = get_network(matrix)
    orc = OllivierRicci(G_nx, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    
    
    for region_count in range(n):
        curvature[region_count] = orc.G.nodes[region_count]['ricciCurvature']
    return curvature

#%%
k = nb_ROI
patient_features = np.zeros((k, patients_count))
printProgressBar(0, patients_count, prefix = 'Progress:', suffix = 'Complete', length = 50)
patient_idx = 0
for patient in patients:
    print(patient)
    G = connectivity_matrices[patient]
    net_resilience = average_neighbor_degree(G)

    patient_features[:, patient_idx] = np.hstack((
        net_resilience
        ))
    
    patient_idx += 1
    printProgressBar(patient_idx, patients_count, prefix = 'Patients progress:', suffix = 'Complete\n', length = 50)

#%% Wang, Ren, Zhang (2017)
#%% Construction of feature vectors
k = nb_ROI * 7 + 2 
patient_features = np.zeros((k, patients_count))
control_features = np.zeros((k, controls_count))
clust_coef_features = []
local_efficiency_features = []
deg_features = []
between_cen_features = []
parti_coef_features = []
net_resilience_features = []
curvature_features = []

printProgressBar(0, patients_count, prefix = 'Progress:', suffix = 'Complete', length = 50)
patient_idx = 0
for patient in patients:
    G = connectivity_matrices[patient]
    clust_coef = clustering_coefficient(G)
    local_efficiency = efficiency(G, local=True)
    charac_path = charpath(G, include_diagonal=True)
    global_efficiency = efficiency(G)
    deg = degree(G)
    between_cen = betweenness_centrality(G)
    ci, _ = modularity_louvain_und(G)
    parti_coef = participation_coef(G, ci)
    net_resilience = average_neighbor_degree(G)
    curvature = node_curvature(G)
    
    clust_coef_features.append(clust_coef)
    local_efficiency_features.append(local_efficiency)
    deg_features.append(deg)
    between_cen_features.append(between_cen)
    parti_coef_features.append(parti_coef)
    net_resilience_features.append(net_resilience)
    curvature_features.append(curvature)
    
    patient_features[:, patient_idx] = np.hstack((
        charac_path, 
        global_efficiency, 
        clust_coef, 
        local_efficiency, 
        deg, 
        between_cen, 
        parti_coef, 
        net_resilience,
        curvature
        ))
    
    patient_idx += 1
    printProgressBar(patient_idx, patients_count, prefix = 'Patients progress:', suffix = 'Complete', length = 50)

printProgressBar(0, controls_count, prefix = 'Progress:', suffix = 'Complete', length = 50)
control_idx = 0
for control in controls:
    G = connectivity_matrices[control]
    clust_coef = clustering_coefficient(G)
    local_efficiency = efficiency(G, local=True)
    charac_path = charpath(G, include_diagonal=True)
    global_efficiency = efficiency(G)
    deg = degree(G)
    between_cen = betweenness_centrality(G)
    ci, _ = modularity_louvain_und(G)
    parti_coef = participation_coef(G, ci)
    net_resilience = average_neighbor_degree(G)
    curvature = node_curvature(G)
    
    clust_coef_features.append(clust_coef)
    local_efficiency_features.append(local_efficiency)
    deg_features.append(deg)
    between_cen_features.append(between_cen)
    parti_coef_features.append(parti_coef)
    net_resilience_features.append(net_resilience)
    curvature_features.append(curvature)

    control_features[:, control_idx] = np.hstack((
        charac_path, 
        global_efficiency, 
        clust_coef, 
        local_efficiency, 
        deg, 
        between_cen, 
        parti_coef, 
        net_resilience,
        curvature
        ))
    
    control_idx += 1
    printProgressBar(control_idx, controls_count, prefix = 'Controls progress:', suffix = 'Complete', length = 50)

subject_features = np.hstack((control_features, patient_features))

print("Feature vectors successfully constructed")

#%% True Feature Selection
def fisher_score(idx):
    FS = [None] * (k - len(idx))
    # list of values to delete from controls and patients 
    controls_values_to_delete = list(filter(lambda x: x<controls_count, idx))
    patients_values_to_delete = list(filter(lambda x: x>=controls_count, idx))
    controls_fs = np.delete(control_features.T, controls_values_to_delete, axis=1)
    patients_fs = np.delete(patient_features.T, patients_values_to_delete, axis=1)
    
    for i in range(k - len(idx)):
        FS[i] = (p1 * (np.mean(controls_fs[:, i]) - np.mean(subject_features.T[:, i])) ** 2 + \
                 p2 * (np.mean(patients_fs[:, i]) - np.mean(subject_features.T[:, i])) ** 2) / \
                (p1 * np.var(controls_fs[:, i]) + p2 * np.var(patients_fs[:, i]))
    
    best_idx_fs = np.argsort(FS)[::-1]
    return best_idx_fs

#%%
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.03, random_state=0) # n_splits = controls_count ? 

y = np.ones((subject_count,))
y[controls_count:] = 0
X = subject_features.T
list_of_best_features = []

printProgressBar(0, cv.get_n_splits(), prefix = 'Progress:', suffix = 'Complete', length = 50)
cnt=0
for train_index, test_index in cv.split(X):
    cnt+=1
    p1 = controls_count
    p2 = patients_count
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    p1 = p1 - sum(idx < controls_count for idx in test_index)
    p2 = p2 - sum(idx >= controls_count for idx in test_index)
    
    assert(p1 + p2 == subject_count - len(test_index))

    best_idx = fisher_score(test_index)[:20]
    acc_prev = 0
    acc = 1
    # increment the number of features to be selected 
    # if the difference of accuracies between current and previous iteration 
    # is very low, break
    '''
    for i in range(1, 50):
        if abs(acc - acc_prev) < 10e-3:
            break
        acc_prev = acc
        selected_features_train = X_train[:, best_idx[:i]]
        selected_features_test = X_test[:, best_idx[:i]]
        cls = SVC(kernel='linear')
        cls.fit(selected_features_train, y_train)
        y_predict = cls.predict(selected_features_test)
        acc = accuracy_score(y_test, y_predict)
    '''
    
    # take 20 best features
    selected_features_train = X_train[:, best_idx]
    selected_features_test = X_test[:, best_idx]
    cls = SVC(kernel='linear')
    cls.fit(selected_features_train, y_train)
    y_predict = cls.predict(selected_features_test)
    acc = accuracy_score(y_test, y_predict)
        
    list_of_best_features.append((best_idx, acc))
    
    printProgressBar(cnt, cv.get_n_splits(X), prefix = 'Patients progress:', suffix = 'Complete', length = 50)

#%% Choose best features among the Shuffle splits
df = pd.DataFrame(list_of_best_features, columns=['features', 'accuracy'])
best_features = []
max_score = max(df['accuracy'])
for i in range(cv.get_n_splits(X)):
    if list_of_best_features[i][1] == max_score:
        best_features.append(list(list_of_best_features[i][0]))
        
best_features2 = []
for i in range(len(best_features)):
    for feature in best_features[i]:
        if feature not in best_features2:
            best_features2.append(feature)
            
#%% Classification with all features

X = subject_features.T[:, best_features2]

cv = ShuffleSplit(n_splits=10, test_size=0.03, random_state=0)
cls = SVC(kernel="linear")
acc = cross_val_score(cls, X, y, cv=cv)
print("Mean accuracy for 10 splits :", np.mean(acc))

#%% Test with permutations 
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score

cls = SVC(kernel="linear", random_state=0)
#cv = StratifiedKFold(2, shuffle=True, random_state=0)

score, perm_scores, pvalue = permutation_test_score(
    cls, X, y, scoring="accuracy", cv=cv, n_permutations=100)

fig, ax = plt.subplots()

ax.hist(perm_scores, bins=20, density=True)
ax.axvline(score, ls="--", color="r")
score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {pvalue:.3f})"
ax.text(0.7, 10, score_label, fontsize=12)
ax.set_xlabel("Accuracy score")
_ = ax.set_ylabel("Probability")
'''
#%% Works in no time
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
cls = SVC(kernel="linear", random_state=0)
cv = StratifiedKFold(2, shuffle=True, random_state=0)

X1 = subject_features.T[:, np.argsort(f_score)[::-1][0:562]]
score, perm_scores, pvalue = permutation_test_score(
    cls, X1, y, scoring="accuracy", cv=cv, n_permutations=1000)

fig, ax = plt.subplots()

ax.hist(perm_scores, bins=20, density=True)
ax.axvline(score, ls="--", color="r")
score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {pvalue:.3f})"
ax.text(0.7, 10, score_label, fontsize=12)
ax.set_xlabel("Accuracy score")
_ = ax.set_ylabel("Probability")
'''
#%% Feature Selection - Fisher score
'''
p1 = len(controls)
p2 = len(patients)

q1 = {}
σ1 = {}
q1['CPL'] = np.mean(patient_features[0, :], axis=0)
q1['GE']= np.mean(patient_features[1, :], axis=0)
σ1['CPL'] = np.var(patient_features[0, :], axis=0)
σ1['GE']= np.var(patient_features[1, :], axis=0)

features = ['CC', 'LE', 'D', 'BC', 'PC', 'AND']
idx = 2
for feature in features:
    q1[feature] = np.mean(patient_features[idx:idx+subject_count, :], axis=0)
    q1[feature] = np.mean(q1[feature])
    σ1[feature] = np.var(patient_features[idx:idx+subject_count, :], axis=0)
    σ1[feature] = np.var(σ1[feature])
    idx += subject_count
    
q2 = {}
σ2 = {}
q2['CPL'] = np.mean(control_features[0, :], axis=0)
q2['GE']= np.mean(control_features[1, :], axis=0)
σ2['CPL'] = np.var(control_features[0, :], axis=0)
σ2['GE']= np.var(control_features[1, :], axis=0)

idx = 2
for feature in features:
    q2[feature] = np.mean(control_features[idx:idx+subject_count, :], axis=0)
    q2[feature] = np.mean(q2[feature])
    σ2[feature] = np.var(patient_features[idx:idx+subject_count, :], axis=0)
    σ2[feature] = np.var(σ2[feature])
    idx += subject_count
    
q = {}    
q['CPL'] = np.mean(subject_features[0, :], axis=0)
q['GE'] = np.mean(subject_features[1, :], axis=0)
idx = 2
for feature in features:
    q[feature] = np.mean(subject_features[idx:idx+subject_count, :], axis=0)
    q[feature] = np.mean(q[feature])
    idx += subject_count
    
FS = {}
features = ['CPL', 'GE', 'CC', 'LE', 'D', 'BC', 'PC', 'AND']
for feature in features:
    FS[feature] = (p1 * (q1[feature] - q[feature]) ** 2 + \
                   p2 * (q2[feature] - q[feature]) ** 2) / \
                   (p1 * σ1[feature] + p2 * σ2[feature])
                   
plt.barh(list(FS.keys()), list(FS.values()))
plt.show()
'''            
