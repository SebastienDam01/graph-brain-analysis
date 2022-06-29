import numpy as np
import scipy as sp
from scipy import stats
from bct import clustering
import matplotlib.pyplot as plt

import argparse
import pickle
import random
from tqdm import tqdm
import copy

import seaborn as sns
sns.set()

def create_arg_parser():
    parser = argparse.ArgumentParser(prog=__file__, description=""" Threshold-free Network-Based Statistics (TFNBS)""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser

def add_arguments(parser):
    parser.add_argument('-m', '--method', type=str, required=False, default='f', help='Method to use to select the threshold. Should be either "f" (for F-test), "welch" or "mannwhitneyu"')
    parser.add_argument('-E', '--E', type=float, required=False, default=0.5, help='Parameter E')
    parser.add_argument('-H', '--H', type=float, required=False, default=2.75, help='Parameter H')
    parser.add_argument('-s', '--nbStep', type=int, required=False, default=100, help='Number of thresholding step interval dh')
    parser.add_argument('-n', '--ntest', type=int, required=False, default=5000, help='Number of permutation tests')
    parser.add_argument('-S', '--save', type=int, required=False, default=1, help='Save plot (1) or not (0)')
    parser.add_argument('-c', '--correction', type=int, required=False, default=1, help='Apply correction for multiple testing or not')
    parser.add_argument('-a', '--alpha', type=float, required=False, default=0.05, help='Alpha')
    
    return parser

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
    parser = create_arg_parser()
    add_arguments(parser)
    args = parser.parse_args()
    
    raw_stats = raw_statistics(x, y, args.method)
    thresholds = range_of_thresholds(raw_stats, args.nbStep)
    tfnbs_matrix = tfnbs(raw_stats, thresholds, args.E, args.H)
    t_max, pval_corrected = permutation_test(x, y, tfnbs_matrix, method=args.method, alpha=args.alpha, ntest=args.ntest, E=args.E, H=args.H)
    
    tfnbs_max_stats =  np.zeros((80, 80))
    tfnbs_max_stats[tfnbs_matrix > t_max] = 1

    if args.correction:
        plt.imshow(tfnbs_max_stats, cmap='gray')
        plt.grid(color='w')
        plt.xlabel('ROIs')
        plt.ylabel('ROIs')
        plt.xticks(np.arange(0, 81, 10))
        plt.yticks(np.arange(0, 81, 10))
        plt.title('TFNBS - {} test - E={} - H={} - \nalpha={} - n_permut={}'.format(args.method, args.E, args.H, args.alpha, args.ntest))
        if args.save:
            plt.savefig('../tfnbs_' + 'H=' + str(args.H) + '_' + str(args.ntest) + '.png')
        plt.show()
        
    else:
        pval_corrected_alpha = copy.deepcopy(pval_corrected)
        pval_corrected_alpha[pval_corrected_alpha > args.alpha] = 1
        pval_corrected_alpha[pval_corrected_alpha == 0] = 1
        np.fill_diagonal(pval_corrected_alpha, 1)
        # dirty
        pval_corrected_alpha_inverse = np.nan_to_num(1 - pval_corrected_alpha)
        plt.imshow(pval_corrected_alpha_inverse, cmap='gray')
        plt.xlabel('ROIs')
        plt.ylabel('ROIs')
        plt.title('TFNBS uncorrected for multiple comparisons - \n {} test - E = {} - H = {} - alpha = {}'.format(args.method, args.E, args.H, args.alpha))
        if args.save:
            plt.savefig('../tfnbs_' + 'H=' + str(args.H) + '_' + str(args.ntest) + '.png')
        plt.show()