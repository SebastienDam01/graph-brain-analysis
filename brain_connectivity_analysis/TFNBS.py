import numpy as np
import scipy as sp
from scipy import stats
from bct import clustering
import matplotlib.pyplot as plt

import argparse
import pickle
import random
from tqdm import tqdm

class TFNBSParamError(RuntimeError):
    pass

class TFnbs():
    def __init__(self, method='f', dh=100, E=0.5, H=2.25):
        self.method = 'f'
        self.dh = dh
        self.E = E
        self.H = H

def f_test(x_, y_):
    f = np.var(x_, ddof=1)/np.var(y_, ddof=1)
    dfn = x_.size - 1 
    dfd = y_.size - 1
    p = 1 - sp.stats.f.cdf(f, dfn, dfd)
    
    return f, p

def raw_statistics(x, y, method='f'):
    nb_ROI = x.shape[0]
    statistics = np.zeros((nb_ROI, nb_ROI))
    
    if method=='welch':
        for i in range(nb_ROI):
            for j in range(i+1, nb_ROI): 
                statistics[i, j], _ = sp.stats.ttest_ind(x[i, j], y[i, j], equal_var=False)
                
    elif method=='f':
        for i in range(nb_ROI):
            for j in range(i+1, nb_ROI): 
                statistics[i, j], _ = f_test(x[i, j], y[i, j])

    elif method=='mannwhitneyu':
        for i in range(nb_ROI):
            for j in range(i+1, nb_ROI): 
                statistics[i, j], _ = sp.stats.mannwhitneyu(x[i, j], y[i, j])
                
    else:
        raise TFNBSParamError("Wrong method chosen. 'welch', 'mannwhitneyu' or 'F'")
                
    # copy upper triangle to lower to obtain symmetric matrix
    statistics = statistics + statistics.T - np.diag(np.diag(statistics))
    
    if (np.isnan(statistics)).sum() > 0:
        statistics = np.nan_to_num(statistics, 0)
    return abs(statistics)

def range_of_thresholds(statistics_, dh=100):
    # s_max = np.max(statistics_) 
    s_max = np.percentile(statistics_, 99.5) # take the 95th percentile because the data fluctuates too much for big values (> 4)
    s_min = np.min(statistics_)
    if s_min==0: # for mannwhitneyu specifically
        s_min = np.min(statistics_[np.where(statistics_ != 0)])
    
    return np.linspace(s_min, s_max, num=dh, endpoint=False)

def tfnbs(statistics, thresh, E=0.5, H=3):
    nb_thresh = len(thresh)
    n = len(statistics)
    scores_matrix = np.zeros((n, n, nb_thresh))
    # only consider upper triangular edges
    ixes = np.where(np.triu(np.ones((n, n)), 1))
    
    # Create the transformed-scores matrix
    for h in range(nb_thresh):
        thresh_stats = np.zeros((n, n))
        
        # threshold
        ind_t, = np.where(list(statistics[np.triu_indices(80, 1)]) > thresh[h])

        # suprathreshold adjacency matrix
        thresh_stats[(ixes[0][ind_t], ixes[1][ind_t])] = 1
        thresh_stats = thresh_stats + thresh_stats.T
            
        # find connected components
        a, sz = clustering.get_components(thresh_stats)

        # replace matrix element value by each component size
        # convert size from nodes to number of edges
        # only consider components comprising more than one node (e.g. a/l 1 edge)
        ind_sz, = np.where(sz > 1)
        ind_sz += 1
        nr_components = np.size(ind_sz)
        sz_links = np.zeros((nr_components,))
        for i in range(nr_components):
            nodes, = np.where(ind_sz[i] == a)
            sz_links[i] = np.sum(thresh_stats[np.ix_(nodes, nodes)]) / 2
            thresh_stats[np.ix_(nodes, nodes)] *= (i + 2)

        # subtract 1 to delete any edges not comprising a component
        thresh_stats[np.where(thresh_stats)] -= 1

        # if np.size(sz_links):
        #     max_sz = np.max(sz_links)
        # else:
        #     # max_sz=0
        #     raise TFNBSParamError('True matrix is degenerate')
        # print('max component size is %i' % max_sz)
        
        # TFCE 
        for comp in range(1, int(np.max(thresh_stats)+1)):
            eh = np.count_nonzero(thresh_stats==comp) / 2
            thresh_stats[np.isin(thresh_stats, comp)] = (eh ** E) * (thresh[h] ** H)
            
        scores_matrix[:, :, h] = thresh_stats
        
    return np.sum(scores_matrix, axis=2)

def permutation_test(x_, y_, mat_obs, alpha, method='f', ntest=1000, E=0.5, H=3):
    xy = np.concatenate((x_, y_), axis=2)
    p = mat_obs.shape[0]
    group_A_count = x_.shape[2]
    group_B_count = y_.shape[2]
    mat_permut = np.zeros((p, p, ntest))
    c = int(np.floor(alpha * ntest))
    thresholds = []
    
    # randomize samples
    print("Computing raw statistics permutation matrices...")
    for t in tqdm(range(ntest)):
        list_A = [random.randint(0, group_A_count-1) for _ in range(group_A_count)]
        list_B = [random.randint(group_A_count, group_A_count + group_B_count -1) for _ in range(group_B_count)]
        
        concat_subset = list_A + list_B
        random.shuffle(concat_subset)
        subset_A, subset_B = concat_subset[:group_A_count], concat_subset[group_A_count:]
        
        mat_permut[:, :, t] = raw_statistics(xy[:, :, subset_A], xy[:, :, subset_B], method=method)
        
        thresholds.append(range_of_thresholds(mat_permut[:, :, t]))
        
    # TFNBS
    print("Computing TFNBS permutation matrices...")
    mat_tfnbs_permut = np.zeros((p, p, ntest))
    for t in tqdm(range(ntest)):
        mat_tfnbs_permut[:, :, t] = tfnbs(mat_permut[:, :, t], thresholds[t], E, H)
    
    # maximal statistic
    max_stat=np.zeros((ntest,))
    for t in range(ntest):
        max_stat[t] = np.max(mat_tfnbs_permut[:, :, t])
    
    # single threshold test
    t_max = np.sort(max_stat)[c]
    print('t_max = {}'.format(t_max))
    
    # unnormalized p-value
    mat_pval = np.zeros((p, p))
    
    print("Computing FWE-corrected edge-wise p-values...")
    for i in range(p):
        for j in range(p):
            # a priori, the number of maximum scores greater than observed scores at [i, j]
            mat_pval[i, j] = np.sum(mat_tfnbs_permut[i, j, :] >= mat_obs[i, j]) / ntest
    
    return t_max, mat_pval

if __name__ == '__main__':
    with open('../manage_data/connection_analysis.pickle', 'rb') as f:
        x, y = pickle.load(f)
    ###
    method='f'
    dh=100
    E=0.5
    H=2.75
    ntest=5000
    ###
    raw_stats = raw_statistics(x, y, method)
    thresholds = range_of_thresholds(raw_stats, dh)
    tfnbs_matrix = tfnbs(raw_stats, thresholds, E, H)
    t_max, pval_corrected = permutation_test(x, y, tfnbs_matrix, method=method, alpha=0.05, ntest=ntest, E=E, H=H)
    
#%% iterations over parameters E, H
tfnbs_matrix_list = []
tfnbs_matrix_stats_list = []
pval_corrected_list = []
test=[]
E_list = [0.5, 0.75]
H_list = np.arange(2.25, 3.55, 0.05)
for E in E_list:
    for H in H_list:
        if E == 0.5:
            if H > 3:
                continue
        if E == 0.75: 
            if H < 2.95:
                continue
        print("H = {:,.2f}".format(H))
        test.append((E,H))
        tfnbs_matrix = tfnbs(raw_stats, thresholds, E, H)
        t_max, pval_corrected = permutation_test(x, y, tfnbs_matrix, method=method, alpha=0.05, ntest=ntest, E=E, H=H)
        
        tfnbs_matrix_list.append(tfnbs_matrix)
        tfnbs_max_stats =  np.zeros((80, 80))# copy.deepcopy(tfnbs_matrix)
        tfnbs_max_stats[tfnbs_matrix > t_max] = 1
        tfnbs_matrix_stats_list.append(tfnbs_max_stats)
        pval_corrected_list.append(pval_corrected)

#%% E=0.5
# https://www.delftstack.com/fr/howto/matplotlib/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/ 
width=20
height=20
rows = 3
cols = 5
         
figure, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height))
plt.subplots_adjust(hspace=-0.7)
figure.suptitle("TFNBS", fontsize=20, y=0.85)

for cnt, b in enumerate(axes.flat):
    b.imshow(tfnbs_matrix_stats_list[cnt])
    b.set_title("H = {:,.2f}".format(H_list[cnt]))
    
figure.tight_layout()
plt.show()

#%% E=0.75
width=20
height=20
rows = 2
cols = 5
         
figure, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height))
plt.subplots_adjust(hspace=-0.7)
figure.suptitle("TFNBS", fontsize=20, y=0.85)

for cnt, b in enumerate(axes.flat):
    b.imshow(tfnbs_matrix_stats_list[16+cnt])
    b.set_title("H = {:,.2f}".format(H_list[16+cnt]))
    
figure.tight_layout()
plt.show()

#%%
# for i in range(dh):
#     plt.imshow(tfnbs_matrix[:,:,i])
#     plt.show()

#%%
# def comp(matrix):
#     n=len(matrix)
#     connected_comp = np.zeros((n, n))
    
#     ixes = np.where(np.triu(np.ones((n, n)), 1))
#     ind_t, = np.where(matrix[np.triu_indices(n, 1)] > 0)
#     print(ind_t)
#     # suprathreshold adjacency matrix
#     connected_comp[(ixes[0][ind_t], ixes[1][ind_t])] = 1
#     connected_comp = connected_comp + connected_comp.T
    
#     plt.imshow(connected_comp)
    
#     a, sz = clustering.get_components(connected_comp)
    
#     # replace matrix element value by each component size
#     # convert size from nodes to number of edges
#     # only consider components comprising more than one node (e.g. a/l 1 edge)
#     ind_sz, = np.where(sz > 1)
#     ind_sz += 1
#     nr_components = np.size(ind_sz)
#     sz_links = np.zeros((nr_components,))
#     for i in range(nr_components):
#         nodes, = np.where(ind_sz[i] == a)
#         sz_links[i] = np.sum(connected_comp[np.ix_(nodes, nodes)]) / 2
#         connected_comp[np.ix_(nodes, nodes)] *= (i + 2)
    
#     # subtract 1 to delete any edges not comprising a component
#     connected_comp[np.where(connected_comp)] -= 1
    
#     return connected_comp

# test=comp(tfnbs_max_stats)

# def comp(matrix):
#     n=len(matrix)
#     copy_matrix = copy.deepcopy(matrix)
#     unique = list(np.unique(copy_matrix))
#     for ind, value in enumerate(unique):
#         for i in range(n):
#             for j in range(n):
#                 if copy_matrix[i, j] == value:
#                     copy_matrix[i, j] = ind
                    
#     return copy_matrix

# test=comp(tfnbs_max_stats)
#%% corrected
import copy 
# import matplotlib as mpl

tfnbs_max_stats =  np.zeros((80, 80))# copy.deepcopy(tfnbs_matrix)
tfnbs_max_stats[tfnbs_matrix > t_max] = 1

# fig, ax = plt.subplots()
# cmap = mpl.cm.get_cmap('tab20', len(np.unique(test)))
# im = plt.imshow(test, cmap=cmap, vmin=0, vmax=len(np.unique(test)), aspect=1, interpolation="none")
# fig.colorbar(im, ticks=range(len(np.unique(test))), orientation="vertical", fraction=0.05, pad=0.04)

plt.imshow(tfnbs_max_stats, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.xticks(np.arange(0, 80, 10))
plt.yticks(np.arange(0, 80, 10))
alpha=0.05
plt.title('TFNBS - {} test - E={} - H={} - \nalpha={} - n_permut={}'.format(method, E, H, alpha, ntest))
plt.savefig('graph_pictures/tfnbs/E=0.5/tfnbs_' + 'H=' + str(H) + '_' + str(ntest) + '.pdf')

#%% plot connectome
from nilearn import plotting

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
    
atlas_region_coords = np.loadtxt('../data/COG_free_80s.txt')

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(tfnbs_max_stats, atlas_region_coords)
disp = plotting.plot_connectome(tfnbs_max_stats, 
                                atlas_threshold,
                                figure=fig)

# disp.savefig('graph_pictures/NBS/' + 'nbs_' + str(threshold_grid[OPTIMAL_THRESHOLD_COUNT]) + '_brain.png', dpi=600)
plotting.show()

#%% uncorrected
alpha=0.05
pval_corrected_alpha = copy.deepcopy(pval_corrected)
pval_corrected_alpha[pval_corrected_alpha > 0.05] = 1
pval_corrected_alpha[pval_corrected_alpha == 0] = 1
np.fill_diagonal(pval_corrected_alpha, 1)
# dirty
pval_corrected_alpha_inverse = np.nan_to_num(1 - pval_corrected_alpha)
plt.imshow(pval_corrected_alpha_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('TFNBS - {} test - E = {} - H = {} - alpha = {}'.format(method, E, H, alpha))
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()

# import seaborn as sns
# sns.boxplot(raw_stats.flatten())
