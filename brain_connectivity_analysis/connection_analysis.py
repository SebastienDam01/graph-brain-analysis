#!/usr/bin/env python3

import pickle
import copy
import sys
import networkx as nx
import statsmodels.api as sm 
from statsmodels.stats.multitest import multipletests
import bct

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from nilearn import plotting
from tqdm import tqdm 
sns.set()

sys.path.append('../utils')
#from utils import printProgressBar

THRESHOLD = 0.3

# Load variables from data_preprocessed.pickle
with open('../manage_data/data_preprocessed.pickle', 'rb') as f:
    connectivity_matrices, controls, patients, controls_count, patients_count, subject_count, patient_info_dict, responders, non_responders, response_df, medication = pickle.load(f)

# Load volumes from volumes_preprocessed.picke
with open('../manage_data/volumes_preprocessed.pickle', 'rb') as f:
    volumes_ROI = pickle.load(f)
    
nb_ROI = len(connectivity_matrices[patients[0]])

# TEMPORARY
patients_to_delete = ['lgp_081LJ',
                      'lgp_096MS',
                      'lgp_086CA',
                       'lgp_115LMR', # exclu
                       'lgp_142JO', # age is NA
                      ] 
controls_to_delete = ['S168',
                     'EMODES_003LS', # no info on excel
                     'EMODES_004ML',
                     'DEP_001SAL', # outliers
                     'DEP_003VB',
                     'DEP_004SC',
                     'DEP_005AS',
                     'DEP_006LD',
                     'DEP_007RT',
                     'DEP_008SR',
                     'DEP_009OP',
                     'DEP_010NL',
                     'DEP_012EP',
                      ]

subjects_to_delete = patients_to_delete + controls_to_delete

for subject in subjects_to_delete:
    if subject in patients:
        patients.remove(subject)
        patients_count -= 1
    else:
        print(subject)
        controls.remove(subject)
        controls_count -= 1
        
subject_count = subject_count - len(subjects_to_delete)

patients.sort()
controls.sort()

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
                mat[i, j] = 2 * mat[i, j] / (volumes[subject][i, 1] + volumes[subject][j, 1])
    
    return densities

def get_adjacency_matrix(E, dim):
    # vcorresp = {k:i for i,k in enumerate(np.unique(E.flatten()))}
    # A = np.zeros((nb_ROI, nb_ROI), dtype=int)
    # for a,b in E:
    #     if a!=b:
    #         A[vcorresp[a],vcorresp[b]] = A[vcorresp[b], vcorresp[a]]=1
            
    A = np.zeros((dim, dim), dtype=int)
    for i, j in E:
        A[i, j] = 1
    return A

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

def threshold_connections(matrices_dict, ratio=0.7):
    matrices_copy = copy.deepcopy(matrices_dict)
    matrices_array = np.zeros((nb_ROI, nb_ROI, subject_count))
    i=0
    keys = []
    for key, mat in matrices_copy.items():
        keys.append(key)
        matrices_array[:, :, i] = mat
        i+=1
    
    counts = (matrices_array == 0).sum(axis=2)
    counts = counts / subject_count
    
    # ind_flat, = np.where(counts[np.triu_indices(80)] < 1-ratio)
    ind = np.argwhere(counts > 1-ratio)
    adj = get_adjacency_matrix(ind, nb_ROI)
    # ixes = np.where(np.triu(np.ones((nb_ROI, nb_ROI)), 1))
    
    for _, mat in matrices_copy.items():
        mat[adj == 1] = 0
        
        # mat[(ixes[0][ind_flat], ixes[1][ind_flat])] = 0
        # mat = mat + mat.T
        
    return ind, matrices_copy

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
    return np.array(data_['Metric'] - (glm_linear_age.fittedvalues - np.mean(glm_linear_age.fittedvalues)))
#%%
# Switcher class for multiple algorithms
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
            print("Running GridSearchCV for %s." % key)
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
            print(k)
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
    
class ParamError(RuntimeError):
    pass
    
#%% Conversion from fiber numbers to density and apply connection threshold
#for patient in connectivity_matrices.keys():
ratio=0.7
connectivity_matrices_wo_threshold = copy.deepcopy(connectivity_matrices)
# connectivity_matrices_wo_threshold = nb_fiber2density(connectivity_matrices_wo_threshold, volumes_ROI)
# count_parsimonious, connectivity_matrices = get_parsimonious_network(connectivity_matrices, ratio=ratio)
null_regions, connectivity_matrices = threshold_connections(connectivity_matrices, ratio=ratio)
null_regions = get_adjacency_matrix(null_regions, dim=nb_ROI)
connectivity_matrices = nb_fiber2density(connectivity_matrices, volumes_ROI)
plt.imshow(null_regions)
plt.title('ratio={}'.format(ratio))

#%% 1. Controls' and patients' connections 
connections_controls = np.zeros((nb_ROI, nb_ROI, controls_count))
connections_patients = np.zeros((nb_ROI, nb_ROI, patients_count))

for control_idx in range(controls_count):
    for i in range(nb_ROI):
        for j in range(nb_ROI):
            connections_controls[i, j, control_idx] = connectivity_matrices[controls[control_idx]][i][j]

for patient_idx in range(patients_count):
    for i in range(nb_ROI):
        for j in range(nb_ROI):
            connections_patients[i, j, patient_idx] = connectivity_matrices[patients[patient_idx]][i][j]
            
#%% Clinical characteristics 
dict_keys = list(patient_info_dict.keys())
for subject in dict_keys:
    if (subject not in patients + controls) or (subject in subjects_to_delete) or (patient_info_dict[subject]['Age']==''):
        print(subject)
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

#%% pickle connections
x=fitted_linear_connections_subjects[:, :, patients_count:] # patients
y=fitted_linear_connections_subjects[:, :, :patients_count] # controls
with open('../manage_data/connection_analysis.pickle', 'wb') as f:
    pickle.dump(
        [x, 
          y], f)
    
#%% 2. t-test
p_value_connection = np.zeros((nb_ROI, nb_ROI))
statistics = np.zeros((nb_ROI, nb_ROI))
for i in tqdm(range(nb_ROI)):
    for j in range(i+1, nb_ROI): 
        statistics[i,j], p_value_connection[i][j], _ = sp.stats.ttest_ind(fitted_linear_connections_subjects[i, j, patients_count:], fitted_linear_connections_subjects[i, j, :patients_count], equal_var=False)

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
# plt.xticks(np.arange(0, nb_ROI + 1, 10))
# plt.yticks(np.arange(0, nb_ROI + 1, 10))
plt.title('Welch test by connections, p < 0.001')
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()

fig = plt.figure(figsize=(6, 2.75))

atlas_threshold = apply_threshold(p_value_connection_bounded_inverse, atlas_region_coords)
disp = plotting.plot_connectome(p_value_connection_bounded_inverse, 
                                atlas_threshold,
                                figure=fig)

#disp.savefig('../brain_connectivity_analysis/graph_pictures/welch_test_brain.png', dpi=300)
plotting.show()

#%% Heatmap 
'''
For large positive t-score, we have evidence that patients mean is greater than the controls mean. 
'''
significant_t_score = copy.deepcopy(statistics)
significant_t_score[p_value_connection_bounded_inverse == 0] = 0
plt.imshow(significant_t_score, cmap='bwr')
plt.colorbar(label="t-statistic")
plt.xticks(np.arange(0, nb_ROI + 1, 10))
plt.yticks(np.arange(0, nb_ROI + 1, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('Heatmap with density of fibers')
#plt.savefig('graph_pictures/heatmap_connection.png', dpi=600)
plt.show()

#%% Retrieve direction of significance in significant ROIs
'''
significant_ROIs = np.argwhere(p_value_connection_bounded_inverse != 0)
# Remove duplicate rows and pairs
dupli_rows = []
for i in range(significant_ROIs.shape[0]):
    if significant_ROIs[i, 0] == significant_ROIs[i, 1]:
        dupli_rows.append(i)
    for j in range(i, significant_ROIs.shape[0]):
        if i!=j and j not in dupli_rows and significant_ROIs[i, 0] == significant_ROIs[j, 1] and significant_ROIs[i, 1] == significant_ROIs[j, 0]:
            dupli_rows.append(j)
            
significant_ROIs = np.delete(significant_ROIs, dupli_rows, 0)

mean_connections_patients = np.mean(connections_patients, axis=2)[significant_ROIs[:, 0], significant_ROIs[:, 1]]
std_connections_patients = np.std(connections_patients, axis=2)[significant_ROIs[:, 0], significant_ROIs[:, 1]]
mean_connections_controls = np.mean(connections_controls, axis=2)[significant_ROIs[:, 0], significant_ROIs[:, 1]]
std_connections_controls = np.std(connections_controls, axis=2)[significant_ROIs[:, 0], significant_ROIs[:, 1]]
'''

#%% FDR
p_value_connection_upper = np.zeros((nb_ROI, nb_ROI))
statistics = np.zeros((nb_ROI, nb_ROI))
for i in tqdm(range(nb_ROI)):
    for j in range(i+1, nb_ROI): 
        statistics[i,j], p_value_connection_upper[i][j], _ = sp.stats.ttest_ind(fitted_linear_connections_subjects[i, j, patients_count:], fitted_linear_connections_subjects[i, j, :patients_count], equal_var=False)

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
p_value_connection_bounded[p_value_connection_bounded > 0.05] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('t-test par connexion, p < 0.05, FDR corrected')
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
p_value_connection_bounded[p_value_connection_bounded > 0.001/(nb_ROI * nb_ROI-1 / 2)] = 1
np.fill_diagonal(p_value_connection_bounded, 1)
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('t-test par connexion, p < 0.001/3160, Bonferroni corrected')
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
x=fitted_linear_connections_subjects[:, :, patients_count:] # patients
y=fitted_linear_connections_subjects[:, :, :patients_count] # controls
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
p_value_connection_bounded[p_value_connection_bounded > 0.05] = 1
# dirty
p_value_connection_bounded_inverse = np.nan_to_num(1 - p_value_connection_bounded)
plt.imshow(p_value_connection_bounded_inverse, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('Shapiro test par connexion, p < 0.05')
#plt.savefig('graph_pictures/ttest_connections_Bonferroni.png', dpi=600)
plt.show()

#%% Histogram 
# for i in range(3160):
#     if len(np.unique(xymat[i, :])) != 1:
#         mu, std = sp.stats.norm.fit(xymat[i,:])
#         xmin, xmax = plt.xlim()
#         absicssa = np.linspace(xmin, xmax, 50)
#         p = sp.stats.norm.pdf(absicssa, mu, std)
#         plt.plot(absicssa, p, 'k', linewidth=2)
#         plt.hist(xymat[i, :], bins=50)
#         plt.title('{}th connection'.format(i))
#         plt.show()

#%% NBS

# proportional threshold, 10 values
# threshold_grid = [0.0005] # for mannwhitneyu test
# threshold_grid = np.arange(0, 80, 4) # for f test
threshold_grid = np.arange(3, 4.1, 0.2) # for quick welch test
# threshold_grid = np.arange(0.4, 4.5, 0.45) # for welch test
# np.arange(0.4, 4.2, 0.4) # for equal variance 
#np.arange(0.45, 4.6, 0.45) # for inequal variance
# np.arange(0.4, 4.5, 0.45) # for density
# [0.01, 0.05, 0.001] # for mannwhitneyu test
nb_grid = len(threshold_grid)
pval_grid, adj_grid = [], []

for thresh_grid in threshold_grid:
    print(len(adj_grid))
    pval, adj, null_K = bct.nbs_bct(x=fitted_linear_connections_subjects[:, :, :patients_count], y=fitted_linear_connections_subjects[:, :, patients_count:], thresh=thresh_grid, method='welch', k=1000)
    pval_grid.append(pval)
    adj_grid.append(adj)

#%%
for i in range(len(adj_grid)):
    plt.imshow(adj_grid[i])
    plt.xticks(np.arange(0, nb_ROI + 1, 10))
    plt.yticks(np.arange(0, nb_ROI + 1, 10))
    plt.xlabel('ROIs')
    plt.ylabel('ROIs')
    plt.title('NBS, threshold={:,.4f}'.format(threshold_grid[i]))
    # plt.savefig('graph_pictures/NBS/' + 'nbs_' + str(threshold_grid[i]) + 'wc.png', dpi=600)
    plt.show()

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
    #'SVC': SVC(),
    #'Ridge': Ridge(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    #'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_jobs=8),
    'BaggingClassifier': BaggingClassifier()
}

params1 = {
    'GaussianBayes': {}, 
    'DecisionTreeClassifier': { 'criterion': ['gini', 'entropy'] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    # 'SVC': [
    #     {'kernel': ['linear'], 'C': [1, 10]},
    #     {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    # ],
    #'Ridge': { 'alpha': [1, 2, 4, 6, 8] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    # 'LogisticRegression': [
    #     { 'solver': ['lbfgs'], 'penalty': ['l2'] },
    #     {'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
    #     {'solver': ['saga'], 'penalty': ['l1', 'l2']},
    #     {'solver': ['saga'], 'penalty': ['elasticnet'], 'l1_ratio': [0.4, 0.8]}
    # ], 
    'KNN': { 'n_neighbors': [2, 3, 5, 7, 9, 11, 13] },
    'BaggingClassifier': { 'n_estimators': [16, 32] }
}

#%% Estimation
results_auc = []
results_f1 = []
for thresh in tqdm(range(nb_grid)):
    for scoring in ['f1', 'roc_auc']:
        # print('Computing scores for {}.'.format(scoring))
        ind = np.where(np.triu(adj_grid[thresh] != 0))
        X = np.zeros((subject_count, len(ind[0])))
        y = np.zeros(subject_count)
        
        for i, (key, mat) in enumerate(connectivity_matrices_wo_threshold.items()):
            X[i, :] = mat[ind]
            y[i] = 0 if key in controls else 1
            
        helper1 = EstimatorSelectionHelper(models1, params1)
        helper1.fit(X, y, scoring=scoring, n_jobs=8, verbose=0)
        
        if scoring == 'f1':
            results_f1.append(helper1.score_summary(sort_by='mean_score'))
        else:
            results_auc.append(helper1.score_summary(sort_by='mean_score'))
            
#%% Plot
# https://stackoverflow.com/questions/71794028/how-can-i-adjust-the-hue-of-a-seaborn-lineplot-without-having-it-connect-to-the
# frame : https://stackoverflow.com/questions/3431nb_ROI + 110/in-pythons-seaborn-is-there-any-way-to-do-the-opposite-of-despine
best_results_auc = pd.DataFrame(index=range(nb_grid), columns=range(3))
best_results_auc.columns = ['Estimator', 'AUC', 'Std']
for i in range(nb_grid):
    best_results_auc.iloc[i, :] = results_auc[i].iloc[0, :3]
best_results_auc['Threshold'] = threshold_grid

fig = plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
ax = sns.lineplot(data=best_results_auc,
                  x='Threshold', 
                  y='AUC',
                  color='black',
                  alpha=0.25
                  )
sns.scatterplot(data=best_results_auc,
             x='Threshold', 
             y='AUC',
             s=40,
             hue='Estimator',
             ax=ax
            )
# plt.errorbar(x=best_results_auc.Threshold, 
#              y=best_results_auc.AUC, 
#              yerr=best_results_auc.Std,
#              color='black',
#              alpha=0.55,
#              ecolor='red',
#              lw=2,
#              capsize=4,
#              capthick=2)
ax.set_facecolor('white')
sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
plt.xticks(threshold_grid)
plt.gca().invert_xaxis()
plt.title('Threshold effect for AUC')

best_results_f1 = pd.DataFrame(index=range(nb_grid), columns=range(3))
best_results_f1.columns = ['Estimator', 'f1', 'Std']
for i in range(nb_grid):
    best_results_f1.iloc[i, :] = results_f1[i].iloc[0, :3]
best_results_f1['Threshold'] = threshold_grid

plt.subplot(1, 2, 2)
#plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=best_results_f1,
                  x='Threshold', 
                  y='f1',
                  color='black',
                  alpha=0.25
                  )
sns.scatterplot(data=best_results_f1,
             x='Threshold', 
             y='f1',
             s=40,
             hue='Estimator',
             ax=ax
            )
# plt.errorbar(x=best_results_f1.Threshold, 
#               y=best_results_f1.f1, 
#               yerr=best_results_f1.Std,
#               color='black',
#               alpha=0.55,
#               ecolor='red',
#               lw=2,
#               capsize=4,
#               capthick=2)
ax.set_facecolor('white')
sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
plt.xticks(threshold_grid)
#plt.ylim(0.55, 1)
plt.gca().invert_xaxis()
plt.title('Threshold effect for f1')
#plt.savefig('graph_pictures/NBS_threshold.png', dpi=600)
# plt.savefig('graph_pictures/NBS_threshold_with_errorbars.png', dpi=600)
plt.show()


#%%
OPTIMAL_THRESHOLD_COUNT = 2

nbs_network = copy.deepcopy(adj_grid[OPTIMAL_THRESHOLD_COUNT])
pval_network = pval_grid[OPTIMAL_THRESHOLD_COUNT]
# Remove subnetworks where number of nodes is two 
# for i in np.unique(nbs_network):
#     if np.count_nonzero((nbs_network) == i) == 2:
#         nbs_network[nbs_network == i] = 0
        
# Remove subnetworks whose p-value is lower than alpha
def remove_subnetworks(pvals, network, alpha=0.1):
    significant_network = copy.deepcopy(network)
    significant_pvals = [idx +1 for idx, pval in enumerate(pvals) if pval < alpha]
    if significant_pvals == []:
        ParamError('No significant edges')
        return None
    else:
        i=1
        for component_nbr in significant_pvals:
            significant_network[significant_network == component_nbr] = i
            i+=1
        
        significant_network[significant_network > len(significant_pvals)] = 0
            
        return significant_network
    
# FIXME
for i in range(nb_ROI):
    for j in range(nb_ROI):
        if np.count_nonzero(nbs_network[i, :]) == 1 and np.count_nonzero(nbs_network[:, j]) == 1:
            nbs_network[i,j]=0
            nbs_network[j,i]=0

nbs_network = remove_subnetworks(pval_network, nbs_network)

# set values starting from 0 and increment by 1 
# in order to have the corresponding colors in cmap

j = 0
for i in np.unique(nbs_network):
    nbs_network[nbs_network == i] = j
    j+=1

fig, ax = plt.subplots()
cmap = mpl.cm.get_cmap('Accent', len(np.unique(nbs_network)))
im = plt.imshow(nbs_network, cmap=cmap, vmin=0, vmax=len(np.unique(nbs_network)), aspect=1, interpolation="none")
fig.colorbar(im, ticks=range(len(np.unique(nbs_network))), orientation="vertical", fraction=0.05, pad=0.04)
plt.xticks(np.arange(0, nb_ROI + 1, 10))
plt.yticks(np.arange(0, nb_ROI + 1, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('NBS, threshold={:,.4f}'.format(threshold_grid[OPTIMAL_THRESHOLD_COUNT]))
#plt.savefig('graph_pictures/NBS/' + 'nbs_' + str(threshold_grid[OPTIMAL_THRESHOLD_COUNT]) + '.png', dpi=300)
plt.show()

threshold_adj = copy.deepcopy(nbs_network)
threshold_adj[threshold_adj != 0] = 1 
degree = threshold_adj @ np.ones(nb_ROI)
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

atlas_threshold = apply_threshold(nbs_network, atlas_region_coords)
atlas_threshold[atlas_threshold==0] = 'nan'
disp = plotting.plot_connectome(nbs_network, 
                                atlas_threshold,
                                node_size=50,
                                # edge_threshold=1,,
                                edge_vmin=0,
                                edge_vmax=len(np.unique(nbs_network)),
                                edge_cmap=cmap,
                                # edge_cmap=mpl.colors.ListedColormap(['red', 'gray', 'yellow', 'green']),
                                # edge_cmap=mpl.colors.ListedColormap(['royalblue', 'dimgray', 'royalblue', 'dimgray', 'royalblue', 'dimgray']),
                                figure=fig)

disp.savefig('graph_pictures/NBS/' + 'nbs_' + str(threshold_grid[OPTIMAL_THRESHOLD_COUNT]) + '_brain.png', dpi=300)
plotting.show()

#%% Export the connections exhibiting significative differences
signif_connections = np.argwhere(nbs_network != 0)
dupli_rows = []
for i in range(signif_connections.shape[0]):
    if signif_connections[i, 0] == signif_connections[i, 1]:
        dupli_rows.append(i)
    for j in range(i, signif_connections.shape[0]):
        if i!=j and j not in dupli_rows and signif_connections[i, 0] == signif_connections[j, 1] and signif_connections[i, 1] == signif_connections[j, 0]:
            dupli_rows.append(j)
            
signif_connections = np.delete(signif_connections, dupli_rows, 0)

X_connections = connections_subjects[signif_connections[:, 0], signif_connections[:, 1], :].T

with open('../manage_data/features_connections.pickle', 'wb') as f:
    pickle.dump(X_connections, f)

#%% Heatmap modified
significant_t_score = copy.deepcopy(statistics)
significant_t_score[nbs_network == 0] = 0
significant_t_score = significant_t_score + significant_t_score.T - np.diag(np.diag(significant_t_score))
plt.imshow(significant_t_score, cmap='bwr')
plt.colorbar(label="t-statistic")
plt.xticks(np.arange(0, nb_ROI + 1, 10))
plt.yticks(np.arange(0, nb_ROI + 1, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
#plt.title('Heatmap of t-score, NBS corrected')
#plt.savefig('graph_pictures/heatmap_connection.png', dpi=300)
plt.show()

#%% BCT ttest
x=fitted_linear_connections_subjects[:, :, patients_count:] # patients
y=fitted_linear_connections_subjects[:, :, :patients_count] # controls
tail="both"
n=nb_ROI
thresh=3

def ttest2_stat_only(x, y, tail):
    t = np.mean(x) - np.mean(y)
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                  * np.var(y, ddof=1)) / (n1 + n2 - 2))
    denom = s * np.sqrt(1 / n1 + 1 / n2)
    # denom = np.sqrt(np.var(x, ddof=1) / n1 + np.var(y, ddof=1) / n2)
    if denom == 0:
        return 0
    if tail == 'both':
        return abs(t / denom)
    if tail == 'left':
        return -t / denom
    else:
        return t / denom

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
for i in range(m):
    t_stat[i] = ttest2_stat_only(xmat[i, :], ymat[i, :], tail)

# threshold
ind_t, = np.where(t_stat > thresh)

# suprathreshold adjacency matrix
adj = np.zeros((n, n))
adj[(ixes[0][ind_t], ixes[1][ind_t])] = 1
# adj[ixes][ind_t]=1
adj = adj + adj.T

plt.imshow(adj, cmap='gray')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.xticks(np.arange(0, nb_ROI + 1, 10))
plt.yticks(np.arange(0, nb_ROI + 1, 10))
plt.title('t-test par connexion, p < 0.001')
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()
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
plt.xticks(np.arange(0, nb_ROI + 1, 10))
plt.yticks(np.arange(0, nb_ROI + 1, 10))
plt.title('t-test par connexion, p < 0.001')
#plt.savefig('brain_connectivity_analysis/graph_pictures_on_good_matrices/ttest_connections.png', dpi=600)
plt.show()

#%% Heatmap
significant_t_score = copy.deepcopy(stats)
significant_t_score[p_value_connection_bounded_inverse == 0] = 0
plt.imshow(significant_t_score, cmap='bwr')
plt.colorbar(label="t-statistic")
plt.xticks(np.arange(0, nb_ROI + 1, 10))
plt.yticks(np.arange(0, nb_ROI + 1, 10))
plt.xlabel('ROIs')
plt.ylabel('ROIs')
# plt.savefig('graph_pictures/heatmap_connection.png', dpi=600)
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
    
    for i, (key, mat) in enumerate(connectivity_matrices_wo_threshold.items()):
        X[i, :] = mat[ind]
        y[i] = 0 if key in controls else 1
    
    cv = StratifiedKFold(n_splits=10)
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        cls = RandomForestClassifier()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        fpr, tpr, threshold_roc = roc_curve(y_test, y_pred, pos_label=1)
        auc_grid[thresh].append(auc(fpr, tpr))
        f1_score_grid[thresh].append(f1_score(y_test, y_pred))
        
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
plt.xlabel('Threshold')
plt.ylabel('Classification score')
plt.title('Classification score for different thresholds')
plt.legend()
plt.show()

#%%
# thresh=4
# ind = np.where(np.triu(adj_grid[thresh] != 0))
# X = np.zeros((subject_count, len(ind[0])))
# y = np.zeros(subject_count)

# for i, (key, mat) in enumerate(connectivity_matrices_wo_threshold.items()):
#     X[i, :] = mat[ind]
#     y[i] = 0 if key in controls else 1

# cv = StratifiedKFold(n_splits=10)
# for train_index, test_index in cv.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
    
#     cls = RandomForestClassifier()
#     cls.fit(X_train, y_train)
#     y_pred = cls.predict(X_test)
#     fpr, tpr, threshold_roc = roc_curve(y_test, y_pred, pos_label=1)
#     auc_grid[thresh].append(auc(fpr, tpr))
#     f1_score_grid[thresh].append(f1_score(y_test, y_pred))

#%% DDT 
x=fitted_linear_connections_subjects[:, :, patients_count:] # patients
y=fitted_linear_connections_subjects[:, :, :patients_count] # controls
n = len(x)

# 1. Difference network 
_, u_pvalue = sp.stats.mannwhitneyu(x, y, axis=-1)
D = 1 - u_pvalue
# np.fill_diagonal(D, 0)
#%% 
# 2. First and second moments
U = 10 # arg
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
    l = μ + np.sqrt(σsq) * np.random.normal(size=(n, m))
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
# quant = np.zeros((U, ))
# for i in range(U):
#     l = μ + np.sqrt(σsq) * np.random.normal(size=(n, m))
#     C[:, :, i] = l @ l.T
#     null[:, :, i] = sp.special.expit(C[:, :, i])
#     quant[i] = np.percentile(C[:, :, i][np.triu_indices(n, 1)], 97.5)
    
# thresh_eDDT = np.exp(np.max(quant)) / (1 + np.exp(np.max(quant)))

# 5. Apply threshold 
γ = sp.special.logit(thresh_aDDT)
A = np.where(D_bar > γ, 1, 0)
d_obs = A @ np.ones(n)

# 6. Generate null distribution for di
sum_A_thresh = np.zeros((n, ))
for u in range(U):
    A_null_thresh = np.where(null[:, :, u] >= thresh_aDDT, 1, 0)
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
