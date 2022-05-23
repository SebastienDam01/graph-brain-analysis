#!/usr/bin/env python3

import pickle
import random
import copy
import sys
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import bct
import statsmodels.api as sm 

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
from utils import printProgressBar

THRESHOLD = 0.3

# Load variables from data_preprocessed.pickle
with open('../manage_data/data_preprocessed.pickle', 'rb') as f:
    connectivity_matrices, controls, patients, controls_count, patients_count, subject_count, patient_info_dict, responders, non_responders = pickle.load(f)

# Load volumes from volumes_preprocessed.picke
with open('../manage_data/volumes_preprocessed.pickle', 'rb') as f:
    volumes_ROI = pickle.load(f)
    
nb_ROI = len(connectivity_matrices[patients[0]])

# TEMPORARY
subjects_to_delete = ['lgp_081LJ',
                      'lgp_096MS',
                      'lgp_086CA',
                      'S168',
                      'EMODES_003LS', # no info on excel
                      'EMODES_004ML']

for subject in subjects_to_delete:
    if subject in patients:
        patients.remove(subject)
        patients_count -= 1
    else:
        print(subject)
        controls.remove(subject)
        controls_count -= 1
        
subject_count = subject_count - len(subjects_to_delete)
# patients_count = patients_count - len(subjects_to_delete)

connectivity_matrices = dict([(key, val) for key, val in 
           connectivity_matrices.items() if key not in subjects_to_delete])

volumes_ROI = dict([(key, val) for key, val in 
           volumes_ROI.items() if key not in subjects_to_delete])

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

def apply_threshold(input_, atlas):
    '''
    Set values which are lesser than threshold to 0.

    Parameters
    ----------
    input_ : Nx1 or NxN np.ndarray
        p values or squared matrix
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
        indices_set_to_zero = [i for i in range(len(input_)) if input_[i] >= 0.05/80]
        atlas_copy[indices_set_to_zero] = 0
        
        indices_set_to_one = [i for i in range(len(input_)) if input_[i] < 0.05/80]
        matrix = np.zeros((nb_ROI, nb_ROI))
        for index in indices_set_to_one:
            matrix[index][index] = 1
        
        return matrix, atlas_copy
    
def plot_fittedvalues(y_, model):
    plt.plot(y_, label='values')
    plt.plot(model.fittedvalues, label='fitted values')
    plt.axvline(x=patients_count, linestyle='--', color='red', label='Patients/Controls separation')
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
    glm_linear_age = sm.GLM.from_formula('Metric ~ Age + Gender', data_).fit()
    return np.array(data['Metric'] - (glm_linear_age.fittedvalues - np.mean(glm_linear_age.fittedvalues)))

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

#%%
dist_connection = {}
for subject in count_parsimonious.keys():
    dist_connection[subject] = len(count_parsimonious[subject])
#%%
plt.figure(figsize=(4, 5))
df = pd.DataFrame.from_dict(dist_connection, orient='index', columns=['edges_removed'])
df = df.sort_values(by=['edges_removed'], ascending=False)
sns.boxplot(y='edges_removed',
            orient='v',
            data=df)
#%% Distribution of connections thresholded
dist_connection = {}
for subject in count_parsimonious.keys():
    if len(count_parsimonious[subject]) > 100:
        dist_connection[subject] = len(count_parsimonious[subject])

plt.figure(figsize=(15, 5))
df = pd.DataFrame.from_dict(dist_connection, orient='index', columns=['edges_removed'])
df = df.sort_values(by=['edges_removed'], ascending=False)
sns.barplot(x=df.index,
            y=df.edges_removed,
            color='darkblue')
plt.ylabel('Number of edges removed')
plt.xticks(rotation=80)
plt.tight_layout()
plt.title('Distribution of the number of edges removed per subject (> 100)')
# plt.savefig('graph_pictures/distribution_threshold.png', dpi=600)
plt.show()
#%% Distribution of volumes per region
dist_volumes = np.zeros((subject_count, nb_ROI))
for region in range(nb_ROI):
    subject_idx = 0
    for key, mat in volumes_ROI.items():
        dist_volumes[subject_idx, region] = mat[region, 0]
        subject_idx += 1
        
mean_volumes = np.zeros((nb_ROI))
std_volumes = np.zeros((nb_ROI))
for i in range(nb_ROI):
    mean_volumes[i] = np.mean(dist_volumes[:, i])
    std_volumes[i] = np.std(dist_volumes[:, i])
  
plt.figure(figsize=(18, 5))
plt.plot(mean_volumes, marker='o', color='black')
plt.fill_between(np.linspace(0,79,80), 
                 mean_volumes - std_volumes, 
                 mean_volumes + std_volumes,
                 alpha=0.5,
                 color='darkgray',
                 edgecolor='dimgray',
                 linewidth=2)
plt.xticks(np.linspace(0,79,80).astype(int), rotation=70)
plt.ylabel('Volume')
plt.xlabel('ROIs')
plt.grid(False)
plt.show()

#%% Clinical characteristics 
dict_keys = list(patient_info_dict.keys())
for subject in dict_keys:
    if subject not in patients + controls or subject in subjects_to_delete or patient_info_dict[subject]['Age']=='':
        del patient_info_dict[subject]
        
age_patients = [int(i) for i in [patient_info_dict[key]['Age'] for key in patient_info_dict.keys() if key in patients]]
age_controls = [int(i) for i in [patient_info_dict[key]['Age'] for key in patient_info_dict.keys() if key in controls]]
gender_patients = [int(i) for i in [patient_info_dict[key]['Gender'] for key in patient_info_dict.keys() if key in patients]]
gender_controls = [int(i) for i in [patient_info_dict[key]['Gender'] for key in patient_info_dict.keys() if key in controls]]
#depression_duration = [int(i) for i in [patient_info_dict[key]['Duree_maladie'] for key in patient_info_dict.keys() if key in patients]]

print("Number of males in patients:", gender_patients.count(1))
print('Number of females in patients:', gender_patients.count(2))
print("Number of males in controls:", gender_controls.count(1))
print('Number of females in controls:', gender_controls.count(2))

print("Patients - Age mean {:,.2f} and standard deviation {:,.2f}".format(np.mean(age_patients), np.std(age_patients)))
print("Controls - Age mean {:,.2f} and standard deviation {:,.2f}".format(np.mean(age_controls), np.std(age_controls)))

#print("Durée de dépression - Moyenne {:,.2f} et écart-type {:,.2f}".format(np.mean(depression_duration), np.std(depression_duration)))

p_value_clinical = {}
_, p_value_clinical['Age'] = sp.stats.ttest_ind(age_patients, age_controls, permutations=5000, equal_var=False)

gender = gender_patients + gender_controls
subjects = patients + controls

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

metrics_patients = dict((k, []) for k in local_metrics + global_metrics)
metrics_controls = dict((k, []) for k in local_metrics + global_metrics)

# Construction of feature vectors
print("Computing graph measures for patients and controls...")
printProgressBar(0, patients_count, prefix = 'Patients progress:', suffix = 'Complete', length = 50)
patient_idx = 0
for patient in patients:
    G = connectivity_matrices[patient]
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
    
    metrics_patients['clust_coef'].append(clust_coef)
    metrics_patients['local_efficiency'].append(local_efficiency)
    metrics_patients['charac_path'].append(charac_path)
    metrics_patients['global_efficiency'].append(global_efficiency)
    metrics_patients['deg'].append(deg)
    metrics_patients['between_cen'].append(between_cen)
    metrics_patients['parti_coef'].append(parti_coef)
    #metrics_patients['net_resilience'].append(net_resilience)
    metrics_patients['curvature'].append(curvature)
    metrics_patients['global_clust_coef'].append(np.mean(clust_coef))
    metrics_patients['strength'].append(strength)
    metrics_patients['global_strength'].append(np.mean(strength))
    #metrics_patients['small_worldness'].append(small_worldness)
    
    patient_idx += 1
    printProgressBar(patient_idx, patients_count, prefix = 'Patients progress:', suffix = 'Complete', length = 50)

printProgressBar(0, controls_count, prefix = 'Controls progress:', suffix = 'Complete', length = 50)
control_idx = 0
for control in controls:
    G = connectivity_matrices[control]
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
    
    metrics_controls['clust_coef'].append(clust_coef)
    metrics_controls['local_efficiency'].append(local_efficiency)
    metrics_controls['charac_path'].append(charac_path)
    metrics_controls['global_efficiency'].append(global_efficiency)
    metrics_controls['deg'].append(deg)
    metrics_controls['between_cen'].append(between_cen)
    metrics_controls['parti_coef'].append(parti_coef)
    # metrics_controls['net_resilience'].append(net_resilience)
    metrics_controls['curvature'].append(curvature)
    metrics_controls['global_clust_coef'].append(np.mean(clust_coef))
    metrics_controls['strength'].append(strength)
    metrics_controls['global_strength'].append(np.mean(strength))
    #metrics_controls['small_worldness'].append(small_worldness)

    control_idx += 1
    printProgressBar(control_idx, controls_count, prefix = 'Controls progress:', suffix = 'Complete', length = 50)

print("Metrics successfully computed.")
    
#%% Measures for each region
measures_patients = {}
for measure in metrics_patients.keys():
    if measure in local_metrics:
        measures_patients[measure] = np.zeros((patients_count, nb_ROI))
        for patient_count in range(len(metrics_patients[measure])):
            measures_patients[measure][patient_count, :] = metrics_patients[measure][patient_count]

measures_controls = {}
for measure in metrics_controls.keys():
    if measure in local_metrics:
        measures_controls[measure] = np.zeros((controls_count, nb_ROI))
        for control_count in range(len(metrics_controls[measure])):
            measures_controls[measure][control_count, :] = metrics_controls[measure][control_count]
            
#%% Mean and std measures of each region
mean_measures_patients = {} # mean value of the measures per region
std_measures_patients = {} # standard deviation of the measures per region
for measure in measures_patients.keys():
    if measure in local_metrics:
        mean_measures_patients[measure] = np.mean(measures_patients[measure], axis=0)
        std_measures_patients[measure] = np.std(measures_patients[measure], axis=0)
            
mean_measures_controls = {} # mean value of the measures per region
std_measures_controls = {} # standard deviation of the measures per region
for measure in measures_patients.keys():
    if measure in local_metrics:
        mean_measures_controls[measure] = np.mean(measures_controls[measure], axis=0)
        std_measures_controls[measure] = np.std(measures_controls[measure], axis=0)

#%% GLM 
#%% Data construction
connections_controls = np.zeros((nb_ROI, nb_ROI, controls_count))
connections_patients = np.zeros((nb_ROI, nb_ROI, patients_count))
age = age_patients + age_controls

#%% Fit GLM
measures_subjects = {}
for metric in local_metrics:
    measures_subjects[metric] = np.vstack((measures_patients[metric], measures_controls[metric]))
for metric in global_metrics:
    measures_subjects[metric] = np.hstack((metrics_patients[metric], metrics_controls[metric]))

data = pd.DataFrame({
    "Intercept": np.ones(subject_count),
    "Age": age - np.mean(age, axis=0),
    "Gender": gender,
    "Metric": np.zeros(subject_count)
    })

for metric in global_metrics:
    data["Metric"] = measures_subjects[metric]
    measures_subjects[metric] = glm_models(data)

for metric in local_metrics:
    print(metric)
    # measures_subjects[metric] = np.zeros((subject_count, nb_ROI))
    for region in tqdm(range(nb_ROI)):
        data["Metric"] = measures_subjects[metric][:, region]
        measures_subjects[metric][:, region] = glm_models(data)

#%% Measures for each region
measures_patients = {}
measures_controls = {}
for metric in local_metrics:
    measures_patients[metric] = measures_subjects[metric][:patients_count, :]
    measures_controls[metric] = measures_subjects[metric][patients_count:, :]
for metric in global_metrics:
    measures_patients[metric] = measures_subjects[metric][:patients_count]
    measures_controls[metric] = measures_subjects[metric][patients_count:]
    
#%% Mean and std measures of each region
mean_measures_patients = {} # mean value of the measures per region
std_measures_patients = {} # standard deviation of the measures per region
for measure in measures_patients.keys():
    if measure in local_metrics:
        mean_measures_patients[measure] = np.mean(measures_patients[measure], axis=0)
        std_measures_patients[measure] = np.std(measures_patients[measure], axis=0)
            
mean_measures_controls = {} # mean value of the measures per region
std_measures_controls = {} # standard deviation of the measures per region
for measure in measures_patients.keys():
    if measure in local_metrics:
        mean_measures_controls[measure] = np.mean(measures_controls[measure], axis=0)
        std_measures_controls[measure] = np.std(measures_controls[measure], axis=0)
    
#%% Statistical test for each region
p_value_region = {}
for measure in local_metrics:
    p_value_region[measure] = np.zeros((nb_ROI))
    for region_count in range(nb_ROI):
        _, p_value_region[measure][region_count] = sp.stats.ttest_ind(measures_patients[measure][:, region_count], measures_controls[measure][:, region_count], equal_var=False)

_, p_value_region['charac_path'] = sp.stats.ttest_ind(measures_subjects['charac_path'][:patients_count], measures_subjects['charac_path'][patients_count:], permutations=5000, equal_var=False)
_, p_value_region['global_efficiency'] = sp.stats.ttest_ind(measures_subjects['global_efficiency'][:patients_count], measures_subjects['global_efficiency'][patients_count:], permutations=5000, equal_var=False)
_, p_value_region['global_clust_coef'] = sp.stats.ttest_ind(measures_subjects['global_clust_coef'][:patients_count], measures_subjects['global_clust_coef'][patients_count:], permutations=5000, equal_var=False)
_, p_value_region['global_strength'] = sp.stats.ttest_ind(measures_subjects['global_strength'][:patients_count], measures_subjects['global_strength'][patients_count:], permutations=5000, equal_var=False)

for measure in p_value_region.keys():
    if measure in local_metrics:
        print(measure, "- Number of p_value inferior to 0.05/80:", (p_value_region[measure] < 0.05/80).sum())
        
#%% Plot values and significant differences - Local measures
atlas_region_coords = np.loadtxt('../data/COG_free_80s.txt')

measures_networks = ['Clustering coefficient',
                     'Local efficiency',
                     'Degree',
                     'Betweenness centrality',
                     'Participation coefficient',
                     #'Net resilience (average neighbor degree)',
                     'Node curvature',
                     'Strength']

i=0
for measure in mean_measures_controls.keys():
    plt.figure(figsize=(18, 5))
    plt.plot(mean_measures_controls[measure], marker='o', color='darkturquoise', label='controls')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_controls[measure] - std_measures_controls[measure], 
                     mean_measures_controls[measure] + std_measures_controls[measure],
                     alpha=0.25,
                     color='cyan',
                     edgecolor='steelblue',
                     linewidth=2)
    
    plt.plot(mean_measures_patients[measure], marker='o', color='black', label='patients')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_patients[measure] - std_measures_patients[measure], 
                     mean_measures_patients[measure] + std_measures_patients[measure],
                     alpha=0.5,
                     color='darkgray',
                     edgecolor='dimgray',
                     linewidth=2)
    
    for region_count in range(nb_ROI):
        if measure != 'charac_path' and measure != 'global_efficiency':
            # Bonferroni correction
            if p_value_region[measure][region_count] < 0.05/80:
                plt.axvline(x=region_count, linestyle='--', color='red')
    plt.ylabel(measures_networks[i])
    plt.xlabel('Regions of Interest (80 ROIs)')
    plt.title(measures_networks[i] + ' - Welch test' + ' - 0 permutation tests', fontweight='bold', loc='center', fontsize=16)
    plt.xticks(np.linspace(0,79,80).astype(int), rotation=70)
    plt.legend()
    # plt.savefig('graph_pictures/' + measures_networks[i] + '.png', dpi=400)
    plt.show()
    
    fig = plt.figure(figsize=(6, 2.75))
    
    matrix_map, atlas_threshold = apply_threshold(p_value_region[measure], atlas_region_coords)
    disp = plotting.plot_connectome(matrix_map, 
                                    atlas_threshold,
                                    figure=fig)

    # disp.savefig('graph_pictures/' + measures_networks[i] + '_brain', dpi=400)
    plotting.show()
    i+=1

#%% Mann-Whitney U test
stats_measures = {}
p_values_mat = {}
n_permut = 50000
print('Computing Mann-Whitney U test ...')
for measure in local_metrics:
    print(measure)
    stats_measures[measure] = np.zeros((nb_ROI,))
    p_values_mat[measure] = np.zeros((nb_ROI,))
    subset_patients = [random.randint(0, patients_count-1) for _ in range(patients_count)] # shuffle index, actually not needed
    subset_controls = [random.randint(patients_count, subject_count-1) for _ in range(controls_count)] # shuffle index, actually not needed
    
    for region_count in range(nb_ROI):
        stats_measures[measure][region_count], _ = sp.stats.mannwhitneyu(measures_patients[measure][:, region_count], measures_controls[measure][:, region_count])
    
    p_values_mat[measure] = permutation_test(subset_controls,
                                    subset_patients,
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
    subset_patients = [random.randint(0, patients_count-1) for _ in range(patients_count)] # hardcoded number of patients taken for each test
    subset_controls = [random.randint(patients_count, subject_count-1) for _ in range(controls_count)] # hardcoded number of controls taken for each test
    
    stats_measures[measure], _ = sp.stats.mannwhitneyu(measures_patients[measure], measures_controls[measure])

    p_values_mat[measure] = permutation_test_global(subset_controls,
                                    subset_patients,
                                    stats_measures[measure],
                                    measure,
                                    5000)
    
#%% Plot values and significant differences - Local measures
p_value = 0.05/80
i=0
for measure in mean_measures_controls.keys():
    plt.figure(figsize=(18, 5))
    plt.plot(mean_measures_controls[measure], marker='o', color='darkturquoise', label='controls')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_controls[measure] - std_measures_controls[measure], 
                     mean_measures_controls[measure] + std_measures_controls[measure],
                     alpha=0.25,
                     color='cyan',
                     edgecolor='steelblue',
                     linewidth=2)
    
    plt.plot(mean_measures_patients[measure], marker='o', color='black', label='patients')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_patients[measure] - std_measures_patients[measure], 
                     mean_measures_patients[measure] + std_measures_patients[measure],
                     alpha=0.5,
                     color='darkgray',
                     edgecolor='dimgray',
                     linewidth=2)
    
    for region_count in range(nb_ROI):
        if measure != 'charac_path' and measure != 'global_efficiency':
            if p_values_mat[measure][region_count] < p_value:
                plt.axvline(x=region_count, linestyle='--', color='red')
    plt.ylabel(measures_networks[i])
    plt.xlabel('Regions of Interest (80 ROIs)')
    plt.title(measures_networks[i] + ' - Mann-Whitney U test - ' + str(n_permut) + ' permutation tests' + ' - p value=' + '0.05/80', fontweight='bold', loc='center', fontsize=16)
    plt.xticks(np.linspace(0,79,80).astype(int), rotation=70)
    plt.legend()
    plt.savefig('graph_pictures/mann-whitney/pdf/' + str(n_permut) + '/' + measures_networks[i] + '.pdf')
    plt.show()
    
    fig = plt.figure(figsize=(6, 2.75))
    
    matrix_map, atlas_threshold = apply_threshold(p_values_mat[measure], atlas_region_coords)
    
    # remove dot at the center
    atlas_threshold[atlas_threshold==0] = 'nan'
    
    # No significative nodes
    if len(np.unique(matrix_map)) == 1 and len(np.unique(atlas_threshold)) == 1:
        matrix_map, atlas_threshold = np.zeros((0, 0)), np.zeros((0, 3))
    disp = plotting.plot_connectome(matrix_map, 
                                    atlas_threshold,
                                    figure=fig)

    disp.savefig('graph_pictures/mann-whitney/pdf/' + str(n_permut) + '/' + measures_networks[i] + '_brain.pdf')
    plotting.show()
    i+=1
    
#%% Without permutation tests
#%% Mann-Whitney U test
p_values_mat_wo_permut = {}
print('Computing Mann-Whitney U test ...')
for measure in local_metrics:
    p_values_mat_wo_permut[measure] = np.zeros((nb_ROI,))
    for region_count in range(nb_ROI):
        _, p_values_mat_wo_permut[measure][region_count] = sp.stats.mannwhitneyu(measures_patients[measure][:, region_count], measures_controls[measure][:, region_count])
    
#%% 
n_permut = 0
p_value = 0.05/80
i=0
for measure in mean_measures_controls.keys():
    plt.figure(figsize=(18, 5))
    plt.plot(mean_measures_controls[measure], marker='o', color='darkturquoise', label='controls')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_controls[measure] - std_measures_controls[measure], 
                     mean_measures_controls[measure] + std_measures_controls[measure],
                     alpha=0.25,
                     color='cyan',
                     edgecolor='steelblue',
                     linewidth=2)
    
    plt.plot(mean_measures_patients[measure], marker='o', color='black', label='patients')
    plt.fill_between(np.linspace(0,79,80), 
                     mean_measures_patients[measure] - std_measures_patients[measure], 
                     mean_measures_patients[measure] + std_measures_patients[measure],
                     alpha=0.5,
                     color='darkgray',
                     edgecolor='dimgray',
                     linewidth=2)
    
    for region_count in range(nb_ROI):
        if measure != 'charac_path' and measure != 'global_efficiency':
            if p_values_mat_wo_permut[measure][region_count] < p_value:
                plt.axvline(x=region_count, linestyle='--', color='red')
    plt.ylabel(measures_networks[i])
    plt.xlabel('Regions of Interest (80 ROIs)')
    plt.title(measures_networks[i] + ' - Mann-Whitney U test - ' + str(n_permut) + ' permutation tests' + ' - p value=' + '0.05/80', fontweight='bold', loc='center', fontsize=16)
    plt.xticks(np.linspace(0,79,80).astype(int), rotation=70)
    plt.legend()
    # plt.savefig('graph_pictures/mann-whitney/pdf/' + str(n_permut) + '/' + measures_networks[i] + '.pdf')
    plt.show()
    
    fig = plt.figure(figsize=(6, 2.75))
    
    matrix_map, atlas_threshold = apply_threshold(p_values_mat[measure], atlas_region_coords)
    # No significative nodes
    if len(np.unique(matrix_map)) == 1 and len(np.unique(atlas_threshold)) == 1:
        matrix_map, atlas_threshold = np.zeros((0, 0)), np.zeros((0, 3))
    disp = plotting.plot_connectome(matrix_map, 
                                    atlas_threshold,
                                    figure=fig)

    # disp.savefig('graph_pictures/mann-whitney/pdf/' + str(n_permut) + '/' + measures_networks[i] + '_brain.pdf')
    plotting.show()
    i+=1
    
#%% All measures
marked_regions = np.array([1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0,0,0,
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
1,
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
0,
0,
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
1
])

node_size = np.array([
4,
0,
0,
0,
0,
0,
0,
2,
0,
0,
1,
2,
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
2,
0,
0,
0,
4,
0,
0,
2,
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
0,
0,
0,
0,
0,
0,
2,
0,
0,
0,
0,
2,
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
2])

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

disp.title('patients vs. controls')

# disp.savefig('graph_pictures/allmeasures_pvsc.pdf')
plotting.show()
i+=1