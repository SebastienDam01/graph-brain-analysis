#!/usr/bin/env python3

import pickle
import random
import copy
import sys
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import bct
import statsmodels.api as sm 
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from nilearn import plotting
from tqdm import tqdm 
sns.set()

sys.path.append('../utils')
#sys.path.insert(0, '../utils')
#from utils import printProgressBar

THRESHOLD = 0.3

# Load variables from data_preprocessed.pickle
with open('../manage_data/data_preprocessed.pickle', 'rb') as f:
    connectivity_matrices, controls, patients, controls_count, patients_count, subject_count, patient_info_dict, responders, non_responders, response_df, medication = pickle.load(f)

# Load volumes from volumes_preprocessed.picke
with open('../manage_data/volumes_preprocessed.pickle', 'rb') as f:
    volumes_ROI = pickle.load(f)
    
nb_ROI = len(connectivity_matrices[responders[0]])

responders_count = len(responders)
non_responders_count = len(non_responders)
subject_count = responders_count + non_responders_count

# TEMPORARY
subjects_to_delete = ['lgp_168CA' # duration disease is NA
                      ]

for subject in subjects_to_delete:
    if subject in responders:
        responders.remove(subject)
        responders_count -= 1
    else:
        print(subject)
        non_responders.remove(subject)
        non_responders_count -= 1
        
subject_count = subject_count - len(subjects_to_delete)

connectivity_matrices = dict([(key, val) for key, val in 
           connectivity_matrices.items() if key not in subjects_to_delete])

volumes_ROI = dict([(key, val) for key, val in 
           volumes_ROI.items() if key not in subjects_to_delete])


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/34325723#34325723
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Density fibers connectivity matrices
def nb_fiber2density(matrices, volumes):
    n = list(matrices.items())[0][1].shape[0]
    densities = copy.deepcopy(matrices)
    densities = dict((k,v.astype(float)) for k,v in densities.items())
    for subject, mat in densities.items():
        for i in range(n):
            for j in range(n):
                mat[i, j] = mat[i, j] / (volumes[subject][i, 1] + volumes[subject][j, 1])
    
    return densities

def get_parsimonious_network(matrices, ratio=0.7, ratio_fiber=0):
    """
    Make the matrix more parcimonious by deleting edges depending on the `ratio`
    and the `ratio_fiber`. Also knowned as consensus threshold.

    Parameters
    ----------
    connectivity_matrices : dict
        Dictionnary of connectivity matrices.
    ratio : float, optional
        If the proportion of subjects >= ratio does not have an edge [i, j], then 
        delete this edge for all subjects. 
        The default is 0.7.
    ratio_fiber : float, optional
        Ratio of the fibers that connects an edge [i, j]. If None, 
        we only see if an edge exists between i and j or not.
        The default is 0.

    Returns
    -------
    NxN ndarray.
    Parsimonious connectivity matrix.

    """
    n = list(matrices.items())[0][1].shape[0]
    threshold = ratio * len(matrices)
    null_entries_count = np.zeros((n, n))
    matrices_copy = copy.deepcopy(matrices)
    key_cnt = {}
    
    for mat in matrices_copy.values():
        null_entries = np.argwhere(mat == 0)
        for i, j in null_entries:
            null_entries_count[i, j] += 1
    
    for key, mat in matrices_copy.items():
        # mat[null_entries_count >= threshold] = 0
        key_cnt[key] = []
        count = 0
        for i in range(n):
            for j in range(n):
                if null_entries_count[i, j] >= threshold and mat[i,j] != 0:
                    mat[i, j] = 0
                    key_cnt[key].append([i, j])
                    count+=1
    return key_cnt, matrices_copy

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

def filter_weights(matrix, ratio):
    """
        Only keep a fraction of the weights of the matrix, fraction specified 
        via the threshold parameter
    """
    n = matrix.shape[0]
    filtered_matrix = np.zeros_like(matrix)
    total_weight = np.sum(matrix)
    weights_id = sorted([(matrix[i,j],i,j) for i in range(n-1) for j in range(i+1,n)], reverse=True)
    
    filtered_weight = 0
    for (w,i,j) in weights_id:
        filtered_weight += 2*w
        
        if filtered_weight > ratio*total_weight:
            break
        
        filtered_matrix[i,j] = w
        filtered_matrix[j,i] = w
    
    return filtered_matrix

def node_curvature(matrix):
    '''
    Compute node curvature for each node of the matrix input.

    Parameters
    ----------
    matrix : NxN np.ndarray
        connectivity matrix.

    Returns
    -------
    curvature : Nx1 np.ndarray
        node curvature vector.

    '''
    n = len(matrix)
    curvature = np.zeros((n))
    G_nx = get_network(matrix)
    orc = OllivierRicci(G_nx, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    
    for region_count in range(n):
        curvature[region_count] = orc.G.nodes[region_count]['ricciCurvature']
    return curvature

def apply_threshold(input_, atlas, threshold=0.05/80):
    '''
    Set values which are lesser than threshold to 0.

    Parameters
    ----------
    input_ : Nx1 or NxN np.ndarray
        p values or squared matrix
    atlas : TO DO
    threshold : float

    Returns
    -------
    dictionary_copy : NxN np.ndarray
        copy of the matrix where the threshold was applied

    '''
    atlas_copy = copy.deepcopy(atlas)
    
    if len(input_.shape) == 2: # matrix NxN  
        indices_set_to_zero = []
        indices_set_to_zero.append([index for index in range(len(input_)) if (
            np.unique(input_[:, index] == 0)[0])]
            )
        atlas_copy[indices_set_to_zero, :] = 0
    
        return atlas_copy
        
    else: # vector Nx1
        indices_set_to_zero = [i for i in range(len(input_)) if input_[i] >= threshold]
        atlas_copy[indices_set_to_zero] = 0
        
        indices_set_to_one = [i for i in range(len(input_)) if input_[i] < threshold]
        matrix = np.zeros((nb_ROI, nb_ROI))
        for index in indices_set_to_one:
            matrix[index][index] = 1
        
        return matrix, atlas_copy
    
def plot_fittedvalues(y_, model):
    plt.plot(y_, label='values')
    plt.plot(model.fittedvalues, label='fitted values')
    plt.axvline(x=responders_count, linestyle='--', color='red', label='responders/non_responders separation')
    plt.ylabel('Global efficiency')
    plt.xlabel('Subject')
    plt.legend()
    plt.grid(False)
    plt.show()

def glm_models(data_):
    """
    Apply Generalized Linear Model to adjust for confounds.

    Parameters
    ----------
    data_ : pandas DataFrame
        Contains columns for Intercept, confounds and the metric observed (response variable).
        The column containing the response variable shall be named 'Metric'.

    Returns
    -------
    ndarray
        Adjusted values for the response variable.
    """
    glm_linear = sm.GLM.from_formula('Metric ~ Age + Gender + Depression_duration', data_).fit()
    return np.array(data['Metric'] - (glm_linear.fittedvalues - np.mean(glm_linear.fittedvalues)))

def permutation_test(list_A, list_B, mat_obs, measure, ntest=1000):
    """
    Perform permutation tests for graph measures. 

    Parameters
    ----------
    list_A : list
        indices or names of first group.
    list_B : list
        indices or names of second group.
    mat_obs : Nx1 np.ndarray
        observed matrix.
    measure : string
        name of tested measure.
    ntest : int, optional
        number of permutations to perform. The default is 1000.

    Returns
    -------
    mat_pval : Nx1 np.ndarray
        matrix of p-values after permutation.

    """
    p = mat_obs.shape[0]
    mat_permut = np.zeros((p, ntest))
        
    # 1. randomize samples
    for t in range(ntest):
        subset_size = len(list_A)
        concat_subset = list_A + list_B
        random.shuffle(concat_subset)
        subset_A, subset_B = concat_subset[:subset_size], concat_subset[subset_size:]
        
        mat_permut[:, t], _ = sp.stats.mannwhitneyu(measures_subjects[measure][subset_A, :], measures_subjects[measure][subset_B, :])
        
    # 2. unnormalized p-value
    mat_pval = np.zeros((p, ))
    
    for j in range(p):
        mat_pval[j] = np.sum(mat_permut[j, :] >= mat_obs[j]) / ntest
            
    return mat_pval

#%%
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

#%% Conversion from fiber numbers to density and apply connection threshold
#for patient in connectivity_matrices.keys():
#    connectivity_matrices[patient] = filter_weights(connectivity_matrices[patient], THRESHOLD)
#connectivity_matrices_wo_threshold = nb_fiber2density(connectivity_matrices, volumes_ROI)
connectivity_matrices_wo_threshold = copy.deepcopy(connectivity_matrices)
count_parsimonious, connectivity_matrices = get_parsimonious_network(connectivity_matrices, ratio=0.85)
connectivity_matrices = nb_fiber2density(connectivity_matrices, volumes_ROI)

#%% Clinical characteristics 
        
age_responders = [int(i) for i in [patient_info_dict[key]['Age'] for key in patient_info_dict.keys() if key in responders]]
age_non_responders = [int(i) for i in [patient_info_dict[key]['Age'] for key in patient_info_dict.keys() if key in non_responders]]
gender_responders = [int(i) for i in [patient_info_dict[key]['Gender'] for key in patient_info_dict.keys() if key in responders]]
gender_non_responders = [int(i) for i in [patient_info_dict[key]['Gender'] for key in patient_info_dict.keys() if key in non_responders]]
depression_duration = response_df.loc[:, 'duree_ma_M0']
depression_duration = depression_duration.dropna()

print("Number of males in responders:", gender_responders.count(1))
print('Number of females in responders:', gender_responders.count(2))
print("Number of males in non_responders:", gender_non_responders.count(1))
print('Number of females in non_responders:', gender_non_responders.count(2))

print("responders - Age mean {:,.2f} and standard deviation {:,.2f}".format(np.mean(age_responders), np.std(age_responders)))
print("non_responders - Age mean {:,.2f} and standard deviation {:,.2f}".format(np.mean(age_non_responders), np.std(age_non_responders)))

print("Durée de dépression - Moyenne {:,.2f} et écart-type {:,.2f}".format(np.mean(depression_duration), np.std(depression_duration)))

p_value_clinical = {}
_, p_value_clinical['Age'] = sp.stats.ttest_ind(age_responders, age_non_responders, permutations=5000, equal_var=False)

gender = gender_responders + gender_non_responders
subjects = responders + non_responders

df = pd.DataFrame({'Gender' : gender, 'Subject':subjects})
contingency = pd.crosstab(df['Gender'], df['Subject'])
_, p_value_clinical['Gender'], _, _ = sp.stats.chi2_contingency(contingency)

##############################
#%% Graph metrics analysis
##############################
#%% Metrics 

local_metrics = ['clust_coef',
                 'local_efficiency',
                 'deg',
                 'between_cen',
                 'parti_coef',
                 # 'net_resilience', # nan values
                 'curvature',
                 'strength']

global_metrics = ['charac_path',
                  'global_efficiency',
                  'global_clust_coef',
                  'global_strength']
                  #'small_worldness']

metrics_responders = dict((k, []) for k in local_metrics + global_metrics)
metrics_non_responders = dict((k, []) for k in local_metrics + global_metrics)

# Construction of feature vectors
print("Computing graph measures for responders and non_responders...")
printProgressBar(0, responders_count, prefix = 'Responders progress:', suffix = 'Complete', length = 50)
patient_idx = 0
for responder in responders:
    G = connectivity_matrices[responder]
    clust_coef = clustering_coefficient(G)
    local_efficiency = efficiency(G, local=True)
    _, distance_matrix = bct.breadthdist(G)
    charac_path = charpath(distance_matrix, include_diagonal=False, include_infinite=False)
    global_efficiency = efficiency(G)
    deg = degree(G)
    between_cen = betweenness_centrality(G)
    ci, _ = modularity_louvain_und(G)
    parti_coef = participation_coef(G, ci)
    #net_resilience = average_neighbor_degree(G)
    curvature = node_curvature(G)
    strength = bct.strengths_und(G)
    #small_worldness = nx.sigma(get_network(G))
    
    metrics_responders['clust_coef'].append(clust_coef)
    metrics_responders['local_efficiency'].append(local_efficiency)
    metrics_responders['charac_path'].append(charac_path)
    metrics_responders['global_efficiency'].append(global_efficiency)
    metrics_responders['deg'].append(deg)
    metrics_responders['between_cen'].append(between_cen)
    metrics_responders['parti_coef'].append(parti_coef)
    #metrics_responders['net_resilience'].append(net_resilience)
    metrics_responders['curvature'].append(curvature)
    metrics_responders['global_clust_coef'].append(np.mean(clust_coef))
    metrics_responders['strength'].append(strength)
    metrics_responders['global_strength'].append(np.mean(strength))
    #metrics_responders['small_worldness'].append(small_worldness)
    
    patient_idx += 1
    printProgressBar(patient_idx, responders_count, prefix = 'Responders progress:', suffix = 'Complete', length = 50)

printProgressBar(0, non_responders_count, prefix = 'non_responders progress:', suffix = 'Complete', length = 50)
control_idx = 0
for non_responder in non_responders:
    G = connectivity_matrices[non_responder]
    clust_coef = clustering_coefficient(G)
    local_efficiency = efficiency(G, local=True)
    _, distance_matrix = bct.breadthdist(G)
    charac_path = charpath(distance_matrix, include_diagonal=False, include_infinite=False)
    global_efficiency = efficiency(G)
    deg = degree(G)
    between_cen = betweenness_centrality(G)
    ci, _ = modularity_louvain_und(G)
    parti_coef = participation_coef(G, ci)
    #net_resilience = average_neighbor_degree(G)
    curvature = node_curvature(G)
    strength = bct.strengths_und(G)
    #small_worldness = nx.sigma(get_network(G))
    
    metrics_non_responders['clust_coef'].append(clust_coef)
    metrics_non_responders['local_efficiency'].append(local_efficiency)
    metrics_non_responders['charac_path'].append(charac_path)
    metrics_non_responders['global_efficiency'].append(global_efficiency)
    metrics_non_responders['deg'].append(deg)
    metrics_non_responders['between_cen'].append(between_cen)
    metrics_non_responders['parti_coef'].append(parti_coef)
    # metrics_non_responders['net_resilience'].append(net_resilience)
    metrics_non_responders['curvature'].append(curvature)
    metrics_non_responders['global_clust_coef'].append(np.mean(clust_coef))
    metrics_non_responders['strength'].append(strength)
    metrics_non_responders['global_strength'].append(np.mean(strength))
    #metrics_non_responders['small_worldness'].append(small_worldness)

    control_idx += 1
    printProgressBar(control_idx, non_responders_count, prefix = 'non_responders progress:', suffix = 'Complete', length = 50)

print("Metrics successfully computed.")
    
#%% Measures for each region
measures_responders = {}
for measure in metrics_responders.keys():
    if measure in local_metrics:
        measures_responders[measure] = np.zeros((responders_count, nb_ROI))
        for patient_count in range(len(metrics_responders[measure])):
            measures_responders[measure][patient_count, :] = metrics_responders[measure][patient_count]

measures_non_responders = {}
for measure in metrics_non_responders.keys():
    if measure in local_metrics:
        measures_non_responders[measure] = np.zeros((non_responders_count, nb_ROI))
        for control_count in range(len(metrics_non_responders[measure])):
            measures_non_responders[measure][control_count, :] = metrics_non_responders[measure][control_count]
            
#%% Mean and std measures of each region
mean_measures_responders = {} # mean value of the measures per region
std_measures_responders = {} # standard deviation of the measures per region
for measure in measures_responders.keys():
    if measure in local_metrics:
        mean_measures_responders[measure] = np.mean(measures_responders[measure], axis=0)
        std_measures_responders[measure] = np.std(measures_responders[measure], axis=0)
            
mean_measures_non_responders = {} # mean value of the measures per region
std_measures_non_responders = {} # standard deviation of the measures per region
for measure in measures_responders.keys():
    if measure in local_metrics:
        mean_measures_non_responders[measure] = np.mean(measures_non_responders[measure], axis=0)
        std_measures_non_responders[measure] = np.std(measures_non_responders[measure], axis=0)

#%% GLM 
#%% Data construction
connections_non_responders = np.zeros((nb_ROI, nb_ROI, non_responders_count))
connections_responders = np.zeros((nb_ROI, nb_ROI, responders_count))
age = age_responders + age_non_responders

#%% Fit GLM
measures_subjects = {}
for metric in local_metrics:
    measures_subjects[metric] = np.vstack((measures_responders[metric], measures_non_responders[metric]))
for metric in global_metrics:
    measures_subjects[metric] = np.hstack((metrics_responders[metric], metrics_non_responders[metric]))

data = pd.DataFrame({
    "Intercept": np.ones(subject_count),
    "Age": age - np.mean(age, axis=0),
    "Gender": gender,
    "Depression_duration": depression_duration,
    "Metric": np.zeros(subject_count)
    })

for metric in global_metrics:
    data["Metric"] = measures_subjects[metric]
    measures_subjects[metric] = glm_models(data)

for metric in local_metrics:
    print(metric)
    # measures_subjects[metric] = np.zeros((subject_count, nb_ROI))
    for region in range(nb_ROI):
        data["Metric"] = measures_subjects[metric][:, region]
        measures_subjects[metric][:, region] = glm_models(data)

#%% Measures for each region
measures_responders = {}
measures_non_responders = {}
for metric in local_metrics:
    measures_responders[metric] = measures_subjects[metric][:responders_count, :]
    measures_non_responders[metric] = measures_subjects[metric][responders_count:, :]
for metric in global_metrics:
    measures_responders[metric] = measures_subjects[metric][:responders_count]
    measures_non_responders[metric] = measures_subjects[metric][responders_count:]
    
#%% Mean and std measures of each region
mean_measures_responders = {} # mean value of the measures per region
std_measures_responders = {} # standard deviation of the measures per region
for measure in measures_responders.keys():
    if measure in local_metrics:
        mean_measures_responders[measure] = np.mean(measures_responders[measure], axis=0)
        std_measures_responders[measure] = np.std(measures_responders[measure], axis=0)
            
mean_measures_non_responders = {} # mean value of the measures per region
std_measures_non_responders = {} # standard deviation of the measures per region
for measure in measures_responders.keys():
    if measure in local_metrics:
        mean_measures_non_responders[measure] = np.mean(measures_non_responders[measure], axis=0)
        std_measures_non_responders[measure] = np.std(measures_non_responders[measure], axis=0)
    
#%% Statistical test for each region
p_value_region = {}
for measure in local_metrics:
    p_value_region[measure] = np.zeros((nb_ROI))
    for region_count in range(nb_ROI):
        _, p_value_region[measure][region_count] = sp.stats.mannwhitneyu(measures_responders[measure][:, region_count], measures_non_responders[measure][:, region_count])

_, p_value_region['charac_path'] = sp.stats.ttest_ind(measures_subjects['charac_path'][:responders_count], measures_subjects['charac_path'][responders_count:], permutations=5000, equal_var=False)
_, p_value_region['global_efficiency'] = sp.stats.ttest_ind(measures_subjects['global_efficiency'][:responders_count], measures_subjects['global_efficiency'][responders_count:], permutations=5000, equal_var=False)
_, p_value_region['global_clust_coef'] = sp.stats.ttest_ind(measures_subjects['global_clust_coef'][:responders_count], measures_subjects['global_clust_coef'][responders_count:], permutations=5000, equal_var=False)
_, p_value_region['global_strength'] = sp.stats.ttest_ind(measures_subjects['global_strength'][:responders_count], measures_subjects['global_strength'][responders_count:], permutations=5000, equal_var=False)

for measure in p_value_region.keys():
    if measure in local_metrics:
        print(measure, "- Number of p_value inferior to 0.05/80:", (p_value_region[measure] < 0.05).sum())

#%% FDR correction
p_value_fdr_region = {}
res_fdr_region = {}
for measure in local_metrics:
    res_fdr_region[measure], p_value_fdr_region[measure], _, _ = multipletests(p_value_region[measure], alpha=0.05, method='fdr_bh')
        
#%% Plot values and significant differences - Local measures
atlas_region_coords = np.loadtxt('../data/COG_free_80s.txt')

p_value = 0.05
measures_networks = ['Clustering coefficient',
                     'Local efficiency',
                     'Degree',
                     'Betweenness centrality',
                     'Participation coefficient',
                     #'Net resilience (average neighbor degree)',
                     'Node curvature',
                     'Strength']

i=0
for measure in mean_measures_non_responders.keys():
    plt.figure(figsize=(18, 5))
    plt.plot(mean_measures_non_responders[measure], marker='o', color='darkturquoise', label='non_responders')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_non_responders[measure] - std_measures_non_responders[measure], 
                     mean_measures_non_responders[measure] + std_measures_non_responders[measure],
                     alpha=0.25,
                     color='cyan',
                     edgecolor='steelblue',
                     linewidth=2)
    
    plt.plot(mean_measures_responders[measure], marker='o', color='black', label='responders')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_responders[measure] - std_measures_responders[measure], 
                     mean_measures_responders[measure] + std_measures_responders[measure],
                     alpha=0.5,
                     color='darkgray',
                     edgecolor='dimgray',
                     linewidth=2)
    
    for region_count in range(nb_ROI):
        if measure != 'charac_path' and measure != 'global_efficiency':
            # Bonferroni correction
            # if p_value_region[measure][region_count] < p_value:
            #     plt.axvline(x=region_count, linestyle='--', color='red')
            # FDR correction
            if res_fdr_region[measure][region_count]:
                plt.axvline(x=region_count, linestyle='--', color='red')
    plt.ylabel(measures_networks[i])
    plt.xlabel('Regions of Interest (80 ROIs)')
    plt.title(measures_networks[i] + ' - t-test' + ' - 5000 permutation tests', fontweight='bold', loc='center', fontsize=16)
    plt.xticks(np.linspace(0,79,80).astype(int), rotation=70)
    plt.legend()
    # plt.savefig('graph_pictures/' + measures_networks[i] + '.png', dpi=400)
    plt.show()
    
    fig = plt.figure(figsize=(6, 2.75))
    
    matrix_map, atlas_threshold = apply_threshold(p_value_region[measure], atlas_region_coords, p_value)
    disp = plotting.plot_connectome(matrix_map, 
                                    atlas_threshold,
                                    figure=fig)

    # disp.savefig('graph_pictures/' + measures_networks[i] + '_brain', dpi=400)
    plotting.show()
    i+=1

#%% Mann-Whitney U test
stats_measures = {}
p_values_mat = {}
n_permut = 5000
print('Computing Mann-Whitney U test ...')
for measure in local_metrics:
    print(measure)
    stats_measures[measure] = np.zeros((nb_ROI,))
    p_values_mat[measure] = np.zeros((nb_ROI,))
    subset_responders = [random.randint(0, responders_count-1) for _ in range(responders_count)] # shuffle index, actually not needed
    subset_non_responders = [random.randint(responders_count, subject_count-1) for _ in range(non_responders_count)] # shuffle index, actually not needed
    
    for region_count in range(nb_ROI):
        stats_measures[measure][region_count], _ = sp.stats.mannwhitneyu(measures_responders[measure][:, region_count], measures_non_responders[measure][:, region_count])
    
    p_values_mat[measure] = permutation_test(subset_non_responders,
                                    subset_responders,
                                    stats_measures[measure],
                                    measure,
                                    n_permut)
    
#%%
def permutation_test_global(list_A, list_B, mat_obs, measure, ntest=1000):
    """
    Perform permutation tests for global graph measures. 

    Parameters
    ----------
    list_A : list
        indices or names of first group.
    list_B : list
        indices or names of second group.
    mat_obs : Nx1 np.ndarray
        observed matrix.
    measure : string
        name of tested measure.
    ntest : int, optional
        number of permutations to perform. The default is 1000.

    Returns
    -------
    mat_pval : Nx1 np.ndarray
        matrix of p-values after permutation.

    """
    mat_permut = np.zeros((ntest))
        
    # 1. randomize samples
    for t in range(ntest):
        subset_size = len(list_A)
        concat_subset = list_A + list_B
        random.shuffle(concat_subset)
        subset_A, subset_B = concat_subset[:subset_size], concat_subset[subset_size:]
        
        mat_permut[t], _ = sp.stats.mannwhitneyu(measures_subjects[measure][subset_A], measures_subjects[measure][subset_B])

    return np.sum(mat_permut >= mat_obs) / ntest

for measure in global_metrics:
    subset_responders = [random.randint(0, responders_count-1) for _ in range(responders_count)] # hardcoded number of responders taken for each test
    subset_non_responders = [random.randint(responders_count, subject_count-1) for _ in range(non_responders_count)] # hardcoded number of non_responders taken for each test
    
    stats_measures[measure], _ = sp.stats.mannwhitneyu(measures_responders[measure], measures_non_responders[measure])

    p_values_mat[measure] = permutation_test_global(subset_non_responders,
                                    subset_responders,
                                    stats_measures[measure],
                                    measure,
                                    5000)
    
#%% FDR correction
p_values_fdr_mat= {}
res_fdr_mat = {}
for measure in local_metrics:
    res_fdr_mat[measure], p_values_fdr_mat[measure], _, _ = multipletests(p_values_mat[measure], alpha=0.05, method='fdr_bh')
    
#%% Plot values and significant differences - Local measures
pvalue = 0.05
i=0
for measure in mean_measures_non_responders.keys():
    plt.figure(figsize=(18, 5))
    plt.plot(mean_measures_non_responders[measure], marker='o', color='darkturquoise', label='non-responders')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_non_responders[measure] - std_measures_non_responders[measure], 
                     mean_measures_non_responders[measure] + std_measures_non_responders[measure],
                     alpha=0.25,
                     color='cyan',
                     edgecolor='steelblue',
                     linewidth=2)
    
    plt.plot(mean_measures_responders[measure], marker='o', color='black', label='responders')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_responders[measure] - std_measures_responders[measure], 
                     mean_measures_responders[measure] + std_measures_responders[measure],
                     alpha=0.5,
                     color='darkgray',
                     edgecolor='dimgray',
                     linewidth=2)
    
    for region_count in range(nb_ROI):
        if measure != 'charac_path' and measure != 'global_efficiency':
            # Bonferroni correction
            # if p_values_mat[measure][region_count] < p_value:
            #     plt.axvline(x=region_count, linestyle='--', color='red')
            # FDR correction
            if res_fdr_region[measure][region_count]:
                plt.axvline(x=region_count, linestyle='--', color='red')
    plt.ylabel(measures_networks[i])
    plt.xlabel('Regions of Interest (80 ROIs)')
    plt.title(measures_networks[i] + ' - Mann-Whitney U test - ' + str(n_permut) + ' permutation tests' + ' - p-value=' + str(pvalue), fontweight='bold', loc='center', fontsize=16)
    plt.xticks(np.linspace(0,79,80).astype(int), rotation=70)
    plt.legend()
    # plt.savefig('graph_pictures/mann-whitney/response/' + str(n_permut) + '/' + measures_networks[i] + '.pdf')
    plt.show()
    
    fig = plt.figure(figsize=(6, 2.75))
    
    matrix_map, atlas_threshold = apply_threshold(p_values_mat[measure], atlas_region_coords, threshold=pvalue)
    
    # remove dot at the center
    atlas_threshold[atlas_threshold==0] = 'nan'
    
    # No significative nodes
    if len(np.unique(matrix_map)) == 1 and len(np.unique(atlas_threshold)) == 1:
        matrix_map, atlas_threshold = np.zeros((0, 0)), np.zeros((0, 3))
    disp = plotting.plot_connectome(matrix_map, 
                                    atlas_threshold,
                                    figure=fig)

    # disp.savefig('graph_pictures/mann-whitney/response/' + str(n_permut) + '/' + measures_networks[i] + '_brain.pdf')
    plotting.show()
    i+=1
    
#%% All measures
marked_regions = np.array([0,
0,
0,
0,
0,
1,
0,
0,
1,
0,
0,
0,
0,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
2,
0,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
0,
0,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
0,
0,
0,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
0,
0

])

node_size = copy.deepcopy(marked_regions)

node_size = node_size * 25

graph_matrix = np.zeros((nb_ROI, nb_ROI))
np.fill_diagonal(graph_matrix, marked_regions)

fig = plt.figure(figsize=(6, 2.75))

atlas = copy.deepcopy(atlas_region_coords)

indices_set_to_zero = [i for i in range(len(marked_regions)) if marked_regions[i] == 0]
atlas[indices_set_to_zero] = 0

disp = plotting.plot_connectome(graph_matrix, 
                                atlas,
                                node_size=node_size,
                                node_color='DarkBlue',
                                figure=fig)

disp.title('responders vs. non-responders')

disp.savefig('graph_pictures/allmeasures_rvsnr.pdf')
plotting.show()
i+=1

    
#%%%%%%%%%%% Connection analysis %%%%%%%%%%%%
# FIXME
#%% 1. Controls' and patients' connections 
connections_non_responders = np.zeros((nb_ROI, nb_ROI, non_responders_count))
connections_responders = np.zeros((nb_ROI, nb_ROI, responders_count))

for non_responder_idx in range(non_responders_count):
    for i in range(nb_ROI):
        for j in range(nb_ROI):
            connections_non_responders[i, j, non_responder_idx] = connectivity_matrices[non_responders[non_responder_idx]][i][j]

for responder_idx in range(responders_count):
    for i in range(nb_ROI):
        for j in range(nb_ROI):
            connections_responders[i, j, responder_idx] = connectivity_matrices[responders[responder_idx]][i][j]
            
connections_subjects = np.concatenate((connections_non_responders, connections_responders), axis=2)

#%%
data = pd.DataFrame({
    "Intercept": np.ones(subject_count),
    "Age": age - np.mean(age, axis=0),
    "Gender": gender,
    "Metric": np.zeros(subject_count)
    })
#%%
fitted_linear_connections_subjects = np.zeros((nb_ROI, nb_ROI, subject_count))
for i in tqdm(range(nb_ROI)):
    for j in range(nb_ROI):
        if np.sum(connections_subjects[i, j, :]) != 0:
            data["Metric"] = connections_subjects[i, j, :]
            fitted_linear_connections_subjects[i, j, :] = glm_models(data)

#%% 2. t-test
p_value_connection = np.zeros((nb_ROI, nb_ROI))
statistics = np.zeros((nb_ROI, nb_ROI))
for i in tqdm(range(nb_ROI)):
    for j in range(i+1, nb_ROI): 
        statistics[i,j], p_value_connection[i][j] = sp.stats.ttest_ind(fitted_linear_connections_subjects[i, j, responders_count:], fitted_linear_connections_subjects[i, j, :responders_count], equal_var=False)

# copy upper triangle to lower to obtain symmetric matrix
p_value_connection = p_value_connection + p_value_connection.T - np.diag(np.diag(p_value_connection))
statistics = statistics + statistics.T - np.diag(np.diag(statistics))

#%%
p_value_connection_bounded = copy.deepcopy(p_value_connection)
p_value_connection_bounded[p_value_connection_bounded > 0.01] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('t-test par connexion, p < 0.01')
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(p_value_connection_bounded_inverse, atlas_region_coords)
disp = plotting.plot_connectome(p_value_connection_bounded_inverse, 
                                atlas_threshold,
                                figure=fig)

# disp.savefig('brain_connectivity_analysis/graph_pictures/' + measures_networks[i] + '_brain', dpi=600)
plotting.show()

#%% Heatmap 
'''
For large positive t-score, we have evidence that patients mean is greater than the controls mean. 
'''
significant_t_score = copy.deepcopy(statistics)
significant_t_score[p_value_connection_bounded_inverse == 0] = 0
plt.imshow(significant_t_score, cmap='bwr')
plt.colorbar(label="t-statistic")
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('Heatmap with density of fibers')
#plt.savefig('graph_pictures/heatmap_connection.png', dpi=600)
plt.show()

#%% FDR
p_value_connection_upper = np.zeros((nb_ROI, nb_ROI))
statistics = np.zeros((nb_ROI, nb_ROI))
for i in tqdm(range(nb_ROI)):
    for j in range(i+1, nb_ROI): 
        statistics[i,j], p_value_connection_upper[i][j] = sp.stats.ttest_ind(fitted_linear_connections_subjects[i, j, responders_count:], fitted_linear_connections_subjects[i, j, :responders_count], equal_var=False)

p_vals = p_value_connection_upper.flatten()
p_vals = np.delete(p_vals, np.where(p_vals == 0))
p_vals = np.nan_to_num(p_vals)
res, pvals_fdr_upper, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_i')

ind = np.triu_indices(80, k=1)
pvals_fdr = np.zeros((80,80),float)
pvals_fdr[ind]=pvals_fdr_upper

pvals_fdr = pvals_fdr + pvals_fdr.T - np.diag(np.diag(pvals_fdr))

#%%
p_value_connection_bounded = copy.deepcopy(pvals_fdr)
p_value_connection_bounded[p_value_connection_bounded > 0.01] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('t-test par connexion, p < 0.01, FDR corrected')
# plt.savefig('graph_pictures/ttest_connections_FDR.png', dpi=600)
plt.show()

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(p_value_connection_bounded_inverse, atlas_region_coords)
disp = plotting.plot_connectome(p_value_connection_bounded_inverse, 
                                atlas_threshold,
                                figure=fig)

# disp.savefig('graph_pictures/ttest_connections_FDR_brain.png', dpi=600)
plotting.show()

#%% Bonferroni
p_value_connection_bounded = copy.deepcopy(p_value_connection)
p_value_connection_bounded[p_value_connection_bounded > 0.01/(nb_ROI * nb_ROI-1 / 2)] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('t-test par connexion, p < 0.01/3160, Bonferroni corrected')
#plt.savefig('graph_pictures/ttest_connections_Bonferroni.png', dpi=600)
plt.show()

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(p_value_connection_bounded_inverse, atlas_region_coords)
disp = plotting.plot_connectome(p_value_connection_bounded_inverse, 
                                atlas_threshold,
                                figure=fig)

#disp.savefig('graph_pictures/ttest_connections_Bonferroni_brain.png', dpi=600)
plotting.show()

#%% Test of equal variances
x=fitted_linear_connections_subjects[:, :, responders_count:] # patients
y=fitted_linear_connections_subjects[:, :, :responders_count] # controls
tail="both"
n=nb_ROI

ix, jx, nx = x.shape
iy, jy, ny = y.shape

# only consider upper triangular edges
ixes = np.where(np.triu(np.ones((n, n)), 1))

# number of edges
m = np.size(ixes, axis=1)

# vectorize connectivity matrices for speed
xmat, ymat = np.zeros((m, nx)), np.zeros((m, ny))

for i in range(nx):
    xmat[:, i] = x[:, :, i][ixes].squeeze()
for i in range(ny):
    ymat[:, i] = y[:, :, i][ixes].squeeze()

# perform t-test at each edge
t_stat = np.zeros((m,))
p_value = np.zeros((m,))
for i in range(m):
    t_stat[i], p_value[i] = sp.stats.levene(xmat[i, :], ymat[i, :])

ind = np.triu_indices(80, k=1)
stats_levene = np.zeros((80,80),float)
stats_levene[ind]=t_stat

stats_levene = stats_levene + stats_levene.T

p_value_levene = np.zeros((80,80),float)
p_value_levene[ind]=p_value

p_value_levene = p_value_levene + p_value_levene.T

alpha = 0.05
p_value_levene[p_value_levene > alpha] = 1
plt.imshow(p_value_levene)
plt.grid(False)
plt.title('Test de levene, p = {}'.format(alpha))
plt.show() 

#print('levene statistics :', ftest_stat)
#print('levene pvalue :', ftest_pvalue)

#%% Test of normality 
xymat = np.concatenate((xmat, ymat), axis=1)
# perform t-test at each edge
shapiro_pvalue = np.zeros((m,))
for i in range(m):
    shapiro_pvalue[i] = sp.stats.shapiro(xymat[i, :])[1]
    
ind = np.triu_indices(nb_ROI, k=1)
shapiro_mat = np.zeros((nb_ROI, nb_ROI))
shapiro_mat[ind]=shapiro_pvalue
shapiro_mat = shapiro_mat + shapiro_mat.T

np.fill_diagonal(shapiro_mat, 0)
p_value_connection_bounded = copy.deepcopy(shapiro_mat)
p_value_connection_bounded[p_value_connection_bounded > 0.01] = 1
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('Shapiro test par connexion, p < 0.001')
#plt.savefig('graph_pictures/ttest_connections_Bonferroni.png', dpi=600)
plt.show()

#%% NBS

# proportional threshold, 10 values
threshold_grid = [0.05, 0.01, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00004, 0.00003] # for mannwhitneyu test
# np.arange(0.4, 4.2, 0.4) # for equal variance 
#np.arange(0.45, 4.6, 0.45) # for inequal variance
# np.arange(0.4, 4.5, 0.45) # for density
# [0.01, 0.05, 0.001] # for mannwhitneyu test
nb_grid = len(threshold_grid)
pval_grid, adj_grid = [], []

for thresh_grid in threshold_grid:
    print(len(adj_grid))
    pval, adj, null_K = bct.nbs_bct(x=fitted_linear_connections_subjects[:, :, :responders_count], y=fitted_linear_connections_subjects[:, :, responders_count:], thresh=thresh_grid, method='mannwhitneyu', k=1000)
    pval_grid.append(pval)
    adj_grid.append(adj)

#%%
for i in range(len(adj_grid)):
    plt.imshow(adj_grid[i])
    plt.xticks(np.arange(0, 81, 10))
    plt.yticks(np.arange(0, 81, 10))
    plt.xlabel('ROIs')
    plt.ylabel('ROIs')
    plt.title('NBS, threshold={:,.5f}'.format(threshold_grid[i]))
    # plt.savefig('graph_pictures/NBS/' + 'nbs_' + str(threshold_grid[i]) + 'wc.png', dpi=600)
    plt.show()

OPTIMAL_THRESHOLD_COUNT = 8
#%%
nbs_network = adj_grid[OPTIMAL_THRESHOLD_COUNT]
# Remove subnetworks where number of nodes is two 
for i in np.unique(nbs_network):
    if np.count_nonzero((nbs_network) == i) == 2:
        nbs_network[nbs_network == i] = 0

# set values starting from 0 and increment by 1 
# in order to have the corresponding colors in cmap

j = 0
for i in np.unique(nbs_network):
    nbs_network[nbs_network == i] = j
    j+=1
    
fig, ax = plt.subplots()
cmap = mpl.cm.get_cmap('Accent', len(np.unique(adj_grid[OPTIMAL_THRESHOLD_COUNT])))
im = plt.imshow(adj_grid[OPTIMAL_THRESHOLD_COUNT], cmap=cmap, vmin=0, vmax=len(np.unique(adj_grid[OPTIMAL_THRESHOLD_COUNT])), aspect=1, interpolation="none")
fig.colorbar(im, ticks=range(len(np.unique(adj_grid[OPTIMAL_THRESHOLD_COUNT]))), orientation="vertical", fraction=0.05, pad=0.04)
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('NBS, threshold={:,.5f}'.format(threshold_grid[OPTIMAL_THRESHOLD_COUNT]))
# plt.savefig('graph_pictures/NBS/response/' + 'nbs_' + str(threshold_grid[OPTIMAL_THRESHOLD_COUNT]) + '.pdf')
plt.show()

threshold_adj = copy.deepcopy(adj_grid[OPTIMAL_THRESHOLD_COUNT])
threshold_adj[threshold_adj != 0] = 1 
degree = threshold_adj @ np.ones(80)
node_size = degree * 50

# # True degrees
# significant_ROIs = np.argwhere(adj_grid[OPTIMAL_THRESHOLD_COUNT] != 0)
# # Remove duplicate rows and pairs
# dupli_rows = []
# for i in range(significant_ROIs.shape[0]):
#     if significant_ROIs[i, 0] == significant_ROIs[i, 1]:
#         dupli_rows.append(i)
#     for j in range(i, significant_ROIs.shape[0]):
#         if i!=j and j not in dupli_rows and significant_ROIs[i, 0] == significant_ROIs[j, 1] and significant_ROIs[i, 1] == significant_ROIs[j, 0]:
#             dupli_rows.append(j)
# significant_ROIs = np.delete(significant_ROIs, dupli_rows, 0)

# unique, counts = np.unique(significant_ROIs.flatten(), return_counts=True)
# print(dict(zip(unique, counts)))

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(adj_grid[OPTIMAL_THRESHOLD_COUNT], atlas_region_coords)
disp = plotting.plot_connectome(adj_grid[OPTIMAL_THRESHOLD_COUNT], 
                                atlas_threshold,
                                node_size=node_size,
                                edge_cmap='Accent',
                                figure=fig)

# disp.savefig('graph_pictures/NBS/response/' + 'nbs_' + str(threshold_grid[OPTIMAL_THRESHOLD_COUNT]) + '_brain.pdf')
plotting.show()

#%% Heatmap modified
statistics = np.zeros((nb_ROI, nb_ROI))
for i in tqdm(range(nb_ROI)):
    for j in range(i+1, nb_ROI): 
        statistics[i,j], _ = sp.stats.ttest_ind(fitted_linear_connections_subjects[i, j, responders_count:], fitted_linear_connections_subjects[i, j, :responders_count], equal_var=False)

statistics = statistics + statistics.T

significant_t_score = copy.deepcopy(statistics)
significant_t_score[adj_grid[OPTIMAL_THRESHOLD_COUNT] == 0] = 0
significant_t_score = significant_t_score + significant_t_score.T - np.diag(np.diag(significant_t_score))

plt.imshow(significant_t_score, cmap='Blues')
plt.colorbar(label="t-statistic")
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('Heatmap of t-score, NBS corrected')
#plt.savefig('graph_pictures/heatmap_connection.png', dpi=600)
plt.show()

#%% DDT 
x=fitted_linear_connections_subjects[:, :, responders_count:] # patients
y=fitted_linear_connections_subjects[:, :, :responders_count] # controls
n = len(x)

# 1. Difference network 
_, u_pvalue = sp.stats.mannwhitneyu(x, y, axis=-1)
D = 1 - u_pvalue
# np.fill_diagonal(D, 0)
#%% 
# 2. First and second moments
U = 2 # arg
D_bar = sp.special.logit(D)
# D_bar[np.diag_indices(n)[0], np.diag_indices(n)[1]] = np.diag(D) # to verify

## Convert -inf and +inf to random values
idxinf = np.argwhere(D_bar <= -12)
idxsup = np.argwhere(D_bar >= 12)
neg = -12 - 1 * np.random.rand(1)
pos = 12 + 1 * np.random.rand(1)
D_bar[idxinf[:, 0], idxinf[:, 1]] = neg
D_bar[idxsup[:, 0], idxsup[:, 1]] = pos

e_bar = np.mean(D_bar[np.triu_indices(n, 1)]) # mean of off-diagonal elements
v_bar = np.var(D_bar[np.triu_indices(n, 1)]) # variance of off-diagonal elements
e = np.mean(np.diag(D_bar))
m = max(2, np.floor((e_bar ** 2 - e ** 2) / v_bar)) # if min (like in paper), returns negative value
μ = np.sqrt(e_bar/m)
σsq = -(μ ** 2) + np.sqrt(μ ** 4 + (v_bar / m))

# 3. Generate U null Difference Networks
C = np.zeros((n, n, U))
null = np.zeros((n, n ,U))

for i in range(U):
    l = μ + np.sqrt(σsq) * np.random.normal(size=(n, U))
    C[:, :, i] = l @ l.T
    null[:, :, i] = sp.special.expit(C[:, :, i])
    
for i in range(U):
    print("mean of off-diagonal elements: {}, expected value: {}".format(np.mean(C[:, :, i][np.triu_indices(n, 1)]), e_bar))
    print("variance of off-diagonal elements: {}, expected value: {}".format(np.var(C[:, :, i][np.triu_indices(n, 1)]), v_bar))
    print("mean of diagonal elements: {}, expected value: {} \n".format(np.mean(np.diag(C[:, :, i])), e))

# 4. Adaptive threshold  

## 4.1 aDDT
df = m
ncp = m * (4 * (μ ** 2)) / (2 * σsq) # non-centrality parameter
mcon = 2 * σsq / 4 # constant
H = mcon * sp.stats.ncx2.rvs(df, ncp, size=1000000) - mcon * sp.stats.chi2.rvs(df, size=1000000)

ll = np.quantile(H, .975)
thresh_aDDT = sp.special.expit(ll)

## 4.2 eDDT
quant = np.zeros((U, ))
for i in range(U):
    l = μ + np.sqrt(σsq) * np.random.normal(size=(n, U))
    C[:, :, i] = l @ l.T
    null[:, :, i] = sp.special.expit(C[:, :, i])
    quant[i] = np.percentile(C[:, :, i][np.triu_indices(n, 1)], 97.5)
    
thresh_eDDT = np.exp(np.max(quant)) / (1 + np.exp(np.max(quant)))

# 5. Apply threshold 
γ = sp.special.logit(thresh_aDDT)
A = np.where(D_bar > γ, 1, 0)
d_obs = A @ np.ones(n)

# 6. Generate null distribution for di
sum_A_thresh = np.zeros((n, ))
for u in range(U):
    A_null_thresh = np.where(null[:, :, u] > thresh_aDDT, 1, 0)
    sum_A_thresh = sum_A_thresh + A_null_thresh @ np.ones(n)
p_null = (1 / (U * (n - 1))) * sum_A_thresh

d_null = np.random.binomial(n-1, p_null)

# 7. Assess the statistical significance of the number of DWE at each node
result = np.where(d_obs > d_null, 1, 0)

#%% Plot
d_obs[result == 0] = 0
fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(A, atlas_region_coords)
disp = plotting.plot_connectome(A, 
                                atlas_threshold,
                                node_size=d_obs,
                                figure=fig)

# disp.savefig('graph_pictures/NBS/' + 'nbs_' + str(threshold_grid[OPTIMAL_THRESHOLD_COUNT]) + '_brain.png', dpi=600)
plotting.show()

#%% Binomial probability density function under the null
p_obs=sp.stats.binom.pmf(d_obs, n-1, p_null)
# Select regions where the DWE is statistically significant
p_obs[result == 0] = 1
# p values that are greater or equal than 0.05 are discarded
pvalue_DDT = copy.deepcopy(p_obs)
pvalue_DDT[pvalue_DDT >= 0.05] = 1
# Discard regions where the corresponding pvalue is greater or equal than 0.05
d_obs_pvalued = copy.deepcopy(d_obs)
d_obs_pvalued[pvalue_DDT == 1] = 0
# Discard regions incident to less than 3 DWE
d_obs_pvalued[d_obs_pvalued < 3] = 0

# FIXME : Remove edges accordingly in adjacency matrix
A_pvalued = copy.deepcopy(A)
for i in range(n):
    if d_obs_pvalued[i] == 0:
        A_pvalued[i, :] = np.zeros(n)
        # A_pvalued[:, i] = A_pvalued[i, :]
        

# fig = plt.figure(figsize=(6, 2.75))

# atlas_threshold = apply_threshold(A, atlas_region_coords)
# disp = plotting.plot_connectome(A, 
#                                 atlas_threshold,
#                                 node_size=d_obs_pvalued,
#                                 figure=fig)

# # disp.savefig('graph_pictures/NBS/' + 'nbs_' + str(threshold_grid[OPTIMAL_THRESHOLD_COUNT]) + '_brain.png', dpi=600)
# plotting.show()