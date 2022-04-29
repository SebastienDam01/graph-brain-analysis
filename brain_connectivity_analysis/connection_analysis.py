#!/usr/bin/env python3

import pickle
import copy
import sys
import networkx as nx
import statsmodels.api as sm 
from statsmodels.stats.multitest import multipletests

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from nilearn import plotting
from tqdm import tqdm 
sns.set()

sys.path.append('../utils')
from utils import printProgressBar

THRESHOLD = 0.3

# Load variables from data_preprocessed.pickle
with open('../manage_data/data_preprocessed.pickle', 'rb') as f:
    connectivity_matrices, controls, patients, controls_count, patients_count, subject_count, patient_info_dict = pickle.load(f)

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

atlas_region_coords = np.loadtxt('../data/COG_free_80s.txt')

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

def get_parsimonious_network(matrices, ratio=0.7, ratio_fiber=0):
    """
    Make the matrix more parcimonious by deleting edges depending on the `ratio`
    and the `ratio_fiber`. 

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
    
def glm_models(data_):
    """
    TO DO

    Parameters
    ----------
    data_ : TYPE
        DESCRIPTION.
        The column containing the response variable shall be named 'Metric'.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    glm_linear_age = sm.GLM.from_formula('Metric ~ Age + Gender', data_).fit()
    return np.array(data['Metric'] - (glm_linear_age.fittedvalues - np.mean(glm_linear_age.fittedvalues)))
    
#%% Conversion from fiber numbers to density and apply connection threshold
#for patient in connectivity_matrices.keys():
#    connectivity_matrices[patient] = filter_weights(connectivity_matrices[patient], THRESHOLD)
#connectivity_matrices_wo_threshold = nb_fiber2density(connectivity_matrices, volumes_ROI)
connectivity_matrices_wo_threshold = copy.deepcopy(connectivity_matrices)
count_parsimonious, connectivity_matrices = get_parsimonious_network(connectivity_matrices, ratio=0.85)
connectivity_matrices = nb_fiber2density(connectivity_matrices, volumes_ROI)

#%% 1. Controls' and patients' connections 
connections_controls = np.zeros((nb_ROI, nb_ROI, controls_count))
connections_patients = np.zeros((nb_ROI, nb_ROI, patients_count))

for control_idx in range(controls_count):
    for i in range(nb_ROI):
        for j in range(nb_ROI):
            connections_controls[i, j, control_idx] = connectivity_matrices_wo_threshold[controls[control_idx]][i][j]

for patient_idx in range(patients_count):
    for i in range(nb_ROI):
        for j in range(nb_ROI):
            connections_patients[i, j, patient_idx] = connectivity_matrices_wo_threshold[patients[patient_idx]][i][j]
            
#%% Clinical characteristics 
dict_keys = list(patient_info_dict.keys())
for subject in dict_keys:
    if subject not in patients + controls or subject in subjects_to_delete or patient_info_dict[subject]['Age']=='':
        del patient_info_dict[subject]
        
age_patients = [int(i) for i in [patient_info_dict[key]['Age'] for key in patient_info_dict.keys() if key in patients]]
age_controls = [int(i) for i in [patient_info_dict[key]['Age'] for key in patient_info_dict.keys() if key in controls]]
gender_patients = [int(i) for i in [patient_info_dict[key]['Gender'] for key in patient_info_dict.keys() if key in patients]]
gender_controls = [int(i) for i in [patient_info_dict[key]['Gender'] for key in patient_info_dict.keys() if key in controls]]

gender = gender_patients + gender_controls
subjects = patients + controls
age = age_patients + age_controls
connections_subjects = np.concatenate((connections_controls, connections_patients), axis=2)

#%%
data = pd.DataFrame({
    "Intercept": np.ones(subject_count),
    "Age": age - np.mean(age, axis=0),
    "Age_squared": (age - np.mean(age, axis=0)) ** 2,
    "Gender": gender,
    "Metric": np.zeros(subject_count)
    })

fitted_linear_connections_subjects = np.zeros((nb_ROI, nb_ROI, subject_count))
fitted_quadratic_connections_subjects = np.zeros((nb_ROI, nb_ROI, subject_count))
fitted_intersection_connections_subjects = np.zeros((nb_ROI, nb_ROI, subject_count))
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
        statistics[i,j], p_value_connection[i][j] = sp.stats.ttest_ind(fitted_linear_connections_subjects[i, j, patients_count:], fitted_linear_connections_subjects[i, j, :patients_count], equal_var=False)

# copy upper triangle to lower to obtain symmetric matrix
p_value_connection = p_value_connection + p_value_connection.T - np.diag(np.diag(p_value_connection))
statistics = statistics + statistics.T - np.diag(np.diag(statistics))
#%%
p_value_connection_bounded = copy.deepcopy(p_value_connection)
p_value_connection_bounded[p_value_connection_bounded > 0.001] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('Ttest par connexion, p < 0.001')
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
#plt.savefig('graph_pictures/heatmap_connection.png', dpi=600)
plt.show()

#%% FDR
p_value_connection_upper = np.zeros((nb_ROI, nb_ROI))
statistics = np.zeros((nb_ROI, nb_ROI))
for i in tqdm(range(nb_ROI)):
    for j in range(i+1, nb_ROI): 
        statistics[i,j], p_value_connection_upper[i][j] = sp.stats.ttest_ind(fitted_linear_connections_subjects[i, j, patients_count:], fitted_linear_connections_subjects[i, j, :patients_count], equal_var=False)

p_vals = p_value_connection_upper.flatten()
p_vals = np.delete(p_vals, np.where(p_vals == 0))
p_vals = np.nan_to_num(p_vals)
res, pvals_fdr_upper, _, _ = multipletests(p_vals, alpha=0.001, method='fdr_i')

ind = np.triu_indices(80, k=1)
pvals_fdr = np.zeros((80,80),float)
pvals_fdr[ind]=pvals_fdr_upper

pvals_fdr = pvals_fdr + pvals_fdr.T - np.diag(np.diag(pvals_fdr))

#%%
p_value_connection_bounded = copy.deepcopy(pvals_fdr)
p_value_connection_bounded[p_value_connection_bounded > 0.001] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('t-test par connexion, p < 0.001, FDR corrected')
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(p_value_connection_bounded_inverse, atlas_region_coords)
disp = plotting.plot_connectome(p_value_connection_bounded_inverse, 
                                atlas_threshold,
                                figure=fig)

# disp.savefig('brain_connectivity_analysis/graph_pictures/' + measures_networks[i] + '_brain', dpi=600)
plotting.show()

#%% Bonferroni
p_value_connection_bounded = copy.deepcopy(p_value_connection)
p_value_connection_bounded[p_value_connection_bounded > 0.001/(nb_ROI * nb_ROI-1 / 2)] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('t-test par connexion, p < 0.001/3160, Bonferroni corrected')
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(p_value_connection_bounded_inverse, atlas_region_coords)
disp = plotting.plot_connectome(p_value_connection_bounded_inverse, 
                                atlas_threshold,
                                figure=fig)

# disp.savefig('brain_connectivity_analysis/graph_pictures/' + measures_networks[i] + '_brain', dpi=600)
plotting.show()

#%% NBS
import matplotlib as mpl
import bct

threshold_grid = np.arange(4, 6.1, 0.1)
nb_grid = len(threshold_grid)
pval_grid, adj_grid = [], []
for thresh_grid in threshold_grid:
    pval, adj, null_K = bct.nbs_bct(x=connections_controls, y=connections_patients, thresh=thresh_grid, k=100)
    pval_grid.append(pval)
    adj_grid.append(adj)

#%%
for i in range(len(adj_grid)):
    plt.imshow(adj_grid[i])
    plt.show()
#%%
fig, ax = plt.subplots()
cmap = mpl.cm.get_cmap('Paired', len(np.unique(adj_grid[15])))
im = plt.imshow(adj_grid[15], cmap=cmap, vmin=0, vmax=len(np.unique(adj_grid[15])), aspect=1, interpolation="none")
fig.colorbar(im, ticks=range(len(np.unique(adj[15]))), orientation="horizontal", fraction=0.05, pad=0.18)
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.show()

#%% Threshold selection by classification 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve, f1_score

auc_grid = [[] for i in range(nb_grid)]
f1_score_grid = [[] for i in range(nb_grid)]

for thresh in tqdm(range(nb_grid)):
    ind = np.where(np.triu(adj_grid[thresh] != 0))
    X = np.zeros((subject_count, len(ind[0])))
    y = np.zeros(subject_count)
    
    for i, (key, mat) in enumerate(connectivity_matrices.items()):
        X[i, :] = mat[ind]
        y[i] = 0 if key in controls else 1
    
    cv = StratifiedKFold(n_splits=10)
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        cls = RandomForestClassifier()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        auc_grid[thresh].append(auc(fpr, tpr))
        f1_score_grid[thresh].append(f1_score(y_test, y_pred))
        
OPTIMAL_THRESHOLD_COUNT = 9
#%% Plot 
plt.figure(figsize=(10, 5))
plt.plot(threshold_grid, np.mean(auc_grid, axis=1), label='AUC')
plt.fill_between(threshold_grid, 
                 np.mean(auc_grid, axis=1) - np.std(auc_grid, axis=1), 
                 np.mean(auc_grid, axis=1) + np.std(auc_grid, axis=1),
                 alpha=0.25,
                 color='cyan',
                 edgecolor='steelblue',
                 linewidth=2)
plt.plot(threshold_grid, np.mean(f1_score_grid, axis=1), label='f1 score', color='black')
plt.fill_between(threshold_grid, 
                 np.mean(f1_score_grid, axis=1) - np.std(f1_score_grid, axis=1), 
                 np.mean(f1_score_grid, axis=1) + np.std(f1_score_grid, axis=1),
                 alpha=0.25,
                 color='darkgray',
                 edgecolor='dimgray',
                 linewidth=2)
plt.xticks(threshold_grid)
plt.legend()
plt.show()

#%%
fig, ax = plt.subplots()
cmap = mpl.cm.get_cmap('Paired', len(np.unique(adj_grid[OPTIMAL_THRESHOLD_COUNT])))
im = plt.imshow(adj_grid[OPTIMAL_THRESHOLD_COUNT], cmap=cmap, vmin=0, vmax=len(np.unique(adj_grid[OPTIMAL_THRESHOLD_COUNT])), aspect=1, interpolation="none")
fig.colorbar(im, ticks=range(len(np.unique(adj_grid[OPTIMAL_THRESHOLD_COUNT]))), orientation="horizontal", fraction=0.05, pad=0.18)
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.show()
#%%
plt.imshow(adj_grid[OPTIMAL_THRESHOLD_COUNT], cmap='gray')
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 81, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('NBS, threshold=4.8')
plt.show()

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(adj_grid[OPTIMAL_THRESHOLD_COUNT], atlas_region_coords)
disp = plotting.plot_connectome(adj_grid[OPTIMAL_THRESHOLD_COUNT], 
                                atlas_threshold,
                                figure=fig)

# disp.savefig('brain_connectivity_analysis/graph_pictures/' + measures_networks[i] + '_brain', dpi=600)
plotting.show()

#%% Switcher class for multiple algorithms
# https://www.davidsbatista.net/blog/2018/02/23/model_optimization/
from sklearn.model_selection import GridSearchCV

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=10, n_jobs=6, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            # print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            # print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'mean_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

#%% Multiple algorithms pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,  GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

models1 = {
    'GaussianBayes': GaussianNB(), 
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'SVC': SVC(),
    #'Ridge': Ridge(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_jobs=8),
    'BaggingClassifier': BaggingClassifier()
}

params1 = {
    'GaussianBayes': {}, 
    'DecisionTreeClassifier': { 'criterion': ['gini', 'entropy'] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ],
    #'Ridge': { 'alpha': [1, 2, 4, 6, 8] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'LogisticRegression': [
        { 'solver': ['lbfgs'], 'penalty': ['l2'] },
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
        {'solver': ['saga'], 'penalty': ['l1', 'l2']},
        {'solver': ['saga'], 'penalty': ['elasticnet'], 'l1_ratio': [0.4, 0.8]}
    ], 
    'KNN': { 'n_neighbors': [2, 3, 5, 7, 9, 11, 13] },
    'BaggingClassifier': { 'n_estimators': [16, 32] }
}

#%% Estimation
for scoring in ['f1', 'roc_auc']:
    print('Computing scores for {}.'.format(scoring))
    ind = np.where(np.triu(adj_grid[OPTIMAL_THRESHOLD_COUNT] != 0))
    X = np.zeros((subject_count, len(ind[0])))
    y = np.zeros(subject_count)
    
    for i, (key, mat) in enumerate(connectivity_matrices.items()):
        X[i, :] = mat[ind]
        y[i] = 0 if key in controls else 1
        
    helper1 = EstimatorSelectionHelper(models1, params1)
    helper1.fit(X, y, scoring=scoring, n_jobs=8, verbose=0)
    
    if scoring == 'f1':
        results_f1 = helper1.score_summary(sort_by='mean_score')
    else:
        results_auc = helper1.score_summary(sort_by='mean_score')
            
#%% Plot
best_results_auc = results_auc.drop_duplicates(subset='estimator')
plt.figure(figsize=(10, 5))
sns.lineplot(x='estimator', y='mean_score', data=best_results_auc)
plt.xticks(results_auc.estimator, rotation=70)
plt.show()
#%% BCT ttest

x=connections_controls
y=connections_patients
tail="both"
n=nb_ROI
thresh=5

def ttest2_stat_only(x, y, tail):
    t = np.mean(x) - np.mean(y)
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                 * np.var(y, ddof=1)) / (n1 + n2 - 2))
    denom = s * np.sqrt(1 / n1 + 1 / n2)
    if denom == 0:
        return 0
    if tail == 'both':
        return np.abs(t / denom)
    if tail == 'left':
        return -t / denom
    else:
        return t / denom

def ttest_paired_stat_only(A, B, tail):
    n = len(A - B)
    df = n - 1
    sample_ss = np.sum((A - B)**2) - np.sum(A - B)**2 / n
    unbiased_std = np.sqrt(sample_ss / (n - 1))
    z = np.mean(A - B) / unbiased_std
    t = z * np.sqrt(n)
    if tail == 'both':
        return np.abs(t)
    if tail == 'left':
        return -t
    else:
        return t

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
del x, y

# perform t-test at each edge
t_stat = np.zeros((m,))
for i in range(m):
    t_stat[i] = ttest2_stat_only(xmat[i, :], ymat[i, :], tail)

# threshold
ind_t, = np.where(t_stat > thresh)

# suprathreshold adjacency matrix
adj = np.zeros((n, n))
adj[(ixes[0][ind_t], ixes[1][ind_t])] = 1
# adj[ixes][ind_t]=1
adj = adj + adj.T
#%%
ind = np.triu_indices(80, k=1)
stats = np.zeros((80,80),float)
stats[ind]=t_stat

stats = stats + stats.T

#https://stackoverflow.com/questions/45045802/how-to-do-a-one-tail-pvalue-calculate-in-python
pval = sp.stats.t.cdf(-abs(stats), subject_count-2)*2

p_value_connection_bounded = copy.deepcopy(pval)
p_value_connection_bounded[p_value_connection_bounded > 0.001] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('t-test par connexion, p < 0.001')
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()

#%% DDT
