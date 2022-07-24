#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:05:07 2022

@author: sdam
"""

import numpy as np
import scipy as sp
from scipy import stats
from bct import clustering
import matplotlib.pyplot as plt
from nilearn import plotting

import argparse
import pickle
import random
from tqdm import tqdm

import seaborn as sns
sns.set()

patients_count = 154
controls_count = 87
atlas_region_coords = np.loadtxt('data/COG_free_80s.txt')

def create_arg_parser():
    parser = argparse.ArgumentParser(prog=__file__, description=""" Threshold-free Network-Based Statistics (TFNBS)""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser

def add_arguments(parser):
    parser.add_argument('-m', '--method', type=str, required=False, help='Method to use to select the threshold. Should be either "f" (for F-test), "t" or "mannwhitneyu"')
    parser.add_argument('-E', '--E', type=float, required=False, help='Parameter E')
    parser.add_argument('-H', '--H', type=float, required=False, help='Parameter H')
    parser.add_argument('-s', '--step', type=int, required=False, help='Thresholding step interval dh')
    parser.add_argument('-n', '--number', type=int, required=False, help='Number of permutation tests')
    
    return parser

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

# class TFnbs():
#     def __init__(self, method='f', dh=100, E=0.5, H=2.25):
#         self.method = 'f'
#         self.dh = dh
#         self.E = E
#         self.H = H

class TFNBSParamError(RuntimeError):
    pass

def f_test(x_, y_):
    f = np.var(x_, ddof=1)/np.var(y_, ddof=1)
    dfn = x_.size - 1 
    dfd = y_.size - 1
    p = 1 - sp.stats.f.cdf(f, dfn, dfd)
    
    return f, p

def raw_statistics(x, y, method='t'):
    nb_ROI = x.shape[0]
    statistics = np.zeros((nb_ROI, nb_ROI))
    
    if method=='t':
        # for i in range(nb_ROI):
        #     for j in range(i+1, nb_ROI): 
        #         statistics[i, j], _, _ = sp.stats.ttest_ind(x[i, j], y[i, j], equal_var=False)
        
        ix, jx, nx = x.shape
        iy, jy, ny = y.shape
        
        # only consider upper triangular edges
        ixes = np.where(np.triu(np.ones((nb_ROI, nb_ROI)), 1))
        
        # number of edges
        m = np.size(ixes, axis=1)
        
        # vectorize connectivity matrices for speed
        xmat, ymat = np.zeros((m, nx)), np.zeros((m, ny))
        
        for i in range(nx):
            xmat[:, i] = x[:, :, i][ixes].squeeze()
        for i in range(ny):
            ymat[:, i] = y[:, :, i][ixes].squeeze()
            
        t_stats = np.zeros((m,))
        for i in range(m):
            t_stats[i], _, _ = sp.stats.ttest_ind(xmat[i,:], ymat[i,:], equal_var=False)
            
        ind = np.triu_indices(nb_ROI, 1)
        statistics[ind] = t_stats
                
    elif method=='f':
        for i in range(nb_ROI):
            for j in range(i+1, nb_ROI): 
                statistics[i, j], _ = f_test(x[i, j], y[i, j])

    elif method=='mannwhitneyu':
        U1 = np.zeros((nb_ROI, nb_ROI))
        for i in range(nb_ROI):
            for j in range(i+1, nb_ROI): 
                U1[i, j], _ = sp.stats.mannwhitneyu(x[i, j], y[i, j])
                U2 = patients_count * controls_count - U1
                statistics = np.minimum(U1, U2)
                
    else:
        raise TFNBSParamError("Wrong method chosen. 't', 'mannwhitneyu' or 'f'")
                
    # copy upper triangle to lower to obtain symmetric matrix
    statistics = statistics + statistics.T - np.diag(np.diag(statistics))
    
    if (np.isnan(statistics)).sum() > 0:
        statistics = np.nan_to_num(statistics, 0)
    return abs(statistics)

def range_of_thresholds(statistics_, dh=100):
    # s_max = np.max(statistics_) 
    s_max = np.percentile(statistics_, 100) # take the 95th percentile because the data fluctuates too much for big values (> 4)
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

def permutation_test(x_, y_, mat_obs, alpha, method='t', ntest=1000, E=0.4, H=3):
    xy = np.concatenate((x_, y_), axis=2)
    p = mat_obs.shape[0]
    group_A_count = x_.shape[2]
    group_B_count = y_.shape[2]
    mat_permut = np.zeros((p, p, ntest))
    c = int(np.floor(alpha * ntest))
    thresholds_perm = []
    
    # randomize samples
    print("Computing raw statistics permutation matrices...")
    for t in tqdm(range(ntest)):
        list_A = [random.randint(0, group_A_count-1) for _ in range(group_A_count)]
        list_B = [random.randint(group_A_count, group_A_count + group_B_count -1) for _ in range(group_B_count)]
        
        concat_subset = list_A + list_B
        random.shuffle(concat_subset)
        subset_A, subset_B = concat_subset[:group_A_count], concat_subset[group_A_count:]
        
        mat_permut[:, :, t] = raw_statistics(xy[:, :, subset_A], xy[:, :, subset_B], method=method)
        
        thresholds_perm.append(range_of_thresholds(mat_permut[:, :, t]))
        
    # TFNBS
    print("Computing TFNBS permutation matrices...")
    mat_tfnbs_permut = np.zeros((p, p, ntest))
    for t in tqdm(range(ntest)):
        mat_tfnbs_permut[:, :, t] = tfnbs(mat_permut[:, :, t], thresholds_perm[t], E, H)
    
    # maximal statistic
    max_stat=np.zeros((ntest,))
    for t in range(ntest):
        max_stat[t] = np.max(mat_tfnbs_permut[:, :, t])
    
    # single threshold test
    t_max = np.sort(max_stat)[::-1][c]
    # t_max = np.percentile(np.sort(max_stat), 100 * (1 - alpha))
    print('t_max = {}'.format(t_max))
    
    # p-values
    mat_pval_uncorrected = np.zeros((p, p))
    mat_pval_corrected = np.zeros((p, p))
    
    compare = lambda x, y: (x <= -np.abs(y)) | (x >= np.abs(y))
    
    print("Computing FWE-corrected edge-wise p-values...")
    for i in range(p):
        for j in range(p):
            # mat_pval_uncorrected[i, j] = np.sum(mat_tfnbs_permut[i, j, :] >= mat_obs[i, j]) / ntest
            mat_pval_uncorrected[i, j] = np.sum(compare(mat_tfnbs_permut[i, j, :], mat_obs[i, j])) / ntest
            mat_pval_corrected[i, j] = np.sum(max_stat >= mat_obs[i, j]) / ntest
    
    return t_max, mat_pval_uncorrected, mat_pval_corrected, mat_tfnbs_permut, thresholds_perm, np.sort(max_stat)

if __name__ == '__main__':
    with open('manage_data/connection_analysis.pickle', 'rb') as f:
        x, y = pickle.load(f)
    ###
    parser = create_arg_parser()
    add_arguments(parser)
    args = parser.parse_args()
    
    method='t'
    dh=100
    alpha=0.05
    E=0.4
    H=3
    ntest=1000
    ###
    t_max_list = []
    raw_stats = raw_statistics(x, y, method)
    thresholds = range_of_thresholds(raw_stats, dh)
    tfnbs_matrix = tfnbs(raw_stats, thresholds, E, H)
    t_max, pval_uncorrected, pval_corrected, tfnbs_matrix_permut, thresh_perm, maximum_statistics = permutation_test(x, y, tfnbs_matrix, method=method, alpha=alpha, ntest=ntest, E=E, H=H)
    
    # plt.hist(raw_stats.flatten()[raw_stats.flatten() < 10], bins=100)
    # plt.show()
    
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
        t_max, pval_uncorrected, pval_corrected, _, _, _ = permutation_test(x, y, tfnbs_matrix, method=method, alpha=0.05, ntest=ntest, E=E, H=H)
        
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
figure.suptitle("TFNBS - E=0.5 - n_permutations={}".format(ntest), fontsize=20, y=0.85)

for cnt, b in enumerate(axes.flat):
    b.imshow(tfnbs_matrix_stats_list[cnt])
    b.set_title("H = {:,.2f}".format(H_list[cnt]))
    
figure.tight_layout()
#plt.savefig('graph_pictures/tfnbs/tfnbs_grid_parameters_E=0.5' + '.png', dpi=300)
plt.show()

#%% E=0.75
width=20
height=20
rows = 2
cols = 5
         
figure, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height))
plt.subplots_adjust(hspace=-0.7)
figure.suptitle("TFNBS - E=0.75 - n_permutations={}".format(ntest), fontsize=20, y=0.75)

for cnt, b in enumerate(axes.flat):
    b.imshow(tfnbs_matrix_stats_list[16+cnt])
    b.set_title("H = {:,.2f}".format(H_list[16+cnt]))

figure.tight_layout()
#plt.savefig('graph_pictures/tfnbs/tfnbs_grid_parameters_E=0.75' + '.png', dpi=300)
plt.show()
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
plt.grid(color='w')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
#plt.title('TFNBS - {} test - E={} - H={} - \nalpha={} - n_permut={}'.format(method, E, H, alpha, ntest))
plt.savefig('graph_pictures/tfnbs/E=0.4/tfnbs_' + 'H=' + str(H) + '_' + str(ntest) + '.pdf')
plt.show()

#%% plot connectome
from matplotlib import cm
from nilearn import plotting

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(tfnbs_max_stats, atlas_region_coords)
disp = plotting.plot_connectome(tfnbs_max_stats, 
                                atlas_threshold,
                                node_color='DarkBlue',
                                edge_cmap=cm.tab10,
                                figure=fig)

# remove dot at the center
atlas_threshold[atlas_threshold==0] = 'nan'

disp.savefig('graph_pictures/tfnbs/E=0.4/tfnbs_' + 'H=' + str(H) + str(ntest) + '_brain.pdf')
plotting.show()

#%% Chosen TFNBS 
from matplotlib import cm

plt.imshow(tfnbs_matrix_stats_list[9], cmap='gray')
plt.grid(color='w')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.title('TFNBS - {} test - E={} - H={} - \nalpha={} - n_permut={}'.format(method, 0.5, 2.75, alpha, ntest))
plt.savefig('graph_pictures/tfnbs/E=0.5/tfnbs_' + 'H=' + str(2.75) + '_' + str(1000) + '.png', dpi=300)
plt.show()

threshold_adj = copy.deepcopy(tfnbs_matrix_stats_list[9])
degree = threshold_adj @ np.ones(80)
node_size = degree * 30

fig = plt.figure(figsize=(6, 2.75))
atlas_threshold = apply_threshold(tfnbs_matrix_stats_list[9], atlas_region_coords)
# remove dot at the center
atlas_threshold[atlas_threshold==0] = 'nan'
disp = plotting.plot_connectome(tfnbs_matrix_stats_list[9], 
                                atlas_threshold,
                                node_color='DarkBlue',
                                edge_cmap=cm.tab10,
                                node_size=node_size,
                                figure=fig)

disp.savefig('graph_pictures/tfnbs/E=0.5/' + 'tfnbs_' + 'H=' + str(2.75) + '_' + str(1000) + '_brain.png', dpi=300)
plotting.show()

#%% uncorrected
alpha=0.05
pval_uncorrected_alpha = copy.deepcopy(pval_uncorrected)
pval_uncorrected_alpha[pval_uncorrected_alpha > alpha] = 1
pval_uncorrected_alpha[pval_uncorrected_alpha == 0] = 1
np.fill_diagonal(pval_uncorrected_alpha, 1)
# dirty
pval_uncorrected_alpha_inverse = np.nan_to_num(1 - pval_uncorrected_alpha)
plt.imshow(pval_uncorrected_alpha_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.title('TFNBS uncorrected for multiple comparisons - \n {} test - E = {} - H = {} - alpha = {}'.format(method, E, H, alpha))
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()

# import seaborn as sns
# sns.boxplot(raw_stats.flatten())

#%% plot connectome

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(pval_uncorrected_alpha_inverse, atlas_region_coords)
disp = plotting.plot_connectome(pval_uncorrected_alpha_inverse, 
                                atlas_threshold,
                                figure=fig)

# disp.savefig('graph_pictures/NBS/' + 'nbs_' + str(threshold_grid[OPTIMAL_THRESHOLD_COUNT]) + '_brain.png', dpi=600)
plotting.show()

#%% corrected with p-value < 0.05
alpha=0.05
pval_corrected_alpha = copy.deepcopy(pval_corrected)
pval_corrected_alpha[pval_corrected_alpha > alpha] = 1
pval_corrected_alpha[pval_corrected_alpha == 0] = 1
np.fill_diagonal(pval_corrected_alpha, 1)
# dirty
pval_corrected_alpha_inverse = np.nan_to_num(1 - pval_corrected_alpha)
plt.imshow(pval_corrected_alpha_inverse, cmap='gray')

plt.imshow(tfnbs_max_stats, cmap='gray')
plt.grid(color='w')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.title('TFNBS - {} test - E={} - H={} - \nalpha={} - n_permut={}'.format(method, E, H, alpha, ntest))
#plt.savefig('graph_pictures/tfnbs/E=0.5/tfnbs_' + 'H=' + str(H) + '_' + str(ntest) + '.pdf')
plt.show()

#%% Correction with FDR
import mne
_, pval_fdr_corrected = mne.stats.fdr_correction(pval_uncorrected)
pval_fdr_corrected_inverse = 1 - pval_fdr_corrected
pval_fdr_corrected_inverse[pval_fdr_corrected_inverse != 1] = 0

plt.imshow(pval_fdr_corrected_inverse, cmap='gray')
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('FDR correction')
plt.show()

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(pval_fdr_corrected_inverse, atlas_region_coords)
disp = plotting.plot_connectome(pval_fdr_corrected_inverse, 
                                atlas_threshold,
                                node_color='DarkBlue',
                                edge_cmap=cm.tab10,
                                figure=fig)

disp.savefig('graph_pictures/tfnbs/E=0.4/tfnbs_fdr_' + 'H=' + str(H) + str(ntest) + '_brain.pdf')
plotting.show()
#%%
from statsmodels.stats.multitest import multipletests
pval_uncorrected_upper = pval_uncorrected[np.triu_indices(79)]

#_, pval_fdr_corrected = mne.stats.fdr_correction(pval_uncorrected_upper)
res, pvals_fdr_upper, _, _ = multipletests(pval_uncorrected_upper, alpha=0.05, method='fdr_i')