#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os
import csv

# Scientific modules
import numpy as np
import pandas as pd

from scipy.io import loadmat
import re 

import seaborn as sns
sns.set()

# ## Configure paths for data once for all

# +

# Folder in which data is placed
data_folder = '../data/glasser/Diffusion_metrics_Glasser'

#%% Rename old prefix by new prefix of volumes if needed
'''
def renamePrefix(data_folder):
    for filename in os.listdir(data_folder):
        if filename[7] == '_':
            print(filename, filename[:7] + filename[8:])
            os.rename(os.path.join(data_folder, filename), os.path.join(data_folder, filename[:7] + filename[8:]))
            
renamePrefix(data_folder)
'''
#%%

# Suffix at the end of each .mat file; it is specified here in order to select 
# relevant files, as well as to make file name lighter during loading for 
# further operations (such as printing subjects names), since it does not carry 
# additional information.
suffix = '_fiber_number.mat'

# For instance here, with those setting, every ../data/*_fiber_number.mat 
# will be loaded

# Keys used to split data between patients and controls. Subject whose filename 
# contains one of the control_keys will be affected to the control cohort, and 
# similarly for patients.
control_keys = ['060', 'dep', 'dpr', 'S', 'TI', 'DEP']
patient_keys = ['lgp']

# By default, the code expects a "table.csv" present in data_folder, containing 
# information about the patients, such as their age, the duration of the 
# disease, etc.
csv_path = data_folder + "/table.csv"
# Second table
csv_june_path = data_folder + "/list_june.csv"

# -

# ## Load data

# +
def get_matrix_file_list(data_folder, suffix):
    """ 
        Return the list of files in the folder data_folder ending by suffix 
    """
    file_list = [f for f in os.listdir(data_folder) if f.endswith(suffix)]
    return list(map(lambda x: data_folder + '/' + x, file_list))

def load_matrix_file(file):
    """ 
        Return the matrix loaded from a Matlab data file. Note that the 
        'Measure' key is hardcoded, so you might need to adjust it to your own 
        data.
    """
    return loadmat(file)['Measure']
    

# Create a dictionnary of all the matrices, where each matrix gets associated to 
# the filename of the corresponding .mat file minus the suffix.
connectivity_matrices = {}

for f in get_matrix_file_list(data_folder, suffix):
    connectivity_matrices[f.replace(suffix,'').replace(data_folder + '/','')] = load_matrix_file(f)
    
# Create a dictionnary of metadata for each patient, obtained from the file 
# at csv_path (by default 'data_foler/table.csv')
patient_info_dict = {}

with open(csv_path, 'r') as csv_file:
    metadata = csv.DictReader(csv_file)
    # Each patient is associated to a dictionnary containing all its information
    for row in metadata:
        metadata_dict = {key:row[key] for key in metadata.fieldnames if key != 'Subject'}
        patient_info_dict[row['Subject']] = metadata_dict
     
print("Succesfully loaded {} matrices from {}.".format(len(connectivity_matrices), data_folder))
print("Metadata has been found in {} for {} subjects.".format(csv_path,len(patient_info_dict)))
# -

# ## Glasser - remove suffix

# +
connectivity_matrices = {k[:-8] if 'Glasser' in k else k:v for k,v in connectivity_matrices.items()}

# ## Split Data into Cohorts

# +
# list of controls and patients names
controls = []
patients = []

# The following can be used to limit the number of either controls or patients 
# considered for the study if they are set to some non infinite number.
controls_count = np.inf
patients_count = np.inf

current_control = 0
current_patient = 0

for key in [*connectivity_matrices]:
    # Use patients_keys and control_keys list to classify subject into cohorts
    if any(list(map(lambda x: x in key, patient_keys))) and current_patient < patients_count:
        patients.append(key)
        current_patient += 1
    elif any(list(map(lambda x: x in key, control_keys))) and current_control < controls_count:
        controls.append(key)
        current_control += 1
    else:
        print("Patient {} cannot be classified either as control or patient.".format(key))

controls_count = current_control
patients_count = current_patient

subject_count = len(patients) + len(controls)

print("Classified {} controls and {} patients (total {} subjects)".format(controls_count, patients_count, subject_count))
    

#%% Check matrices
null_matrices = []
diag_matrices = []
for key, mat in connectivity_matrices.items():
    if not np.any(mat):
        null_matrices.append(key)
    if np.any(mat) and np.all(mat == np.diag(np.diagonal(mat))):
        diag_matrices.append(key)
        
non_fonctional_mat = []
for key, mat in connectivity_matrices.items():
    for i in range(80):
        #print(np.unique(mat[i, i]))
        if (np.unique(mat[i, i]) == np.array([0]))[0]:
            non_fonctional_mat.append((key, i))
            
print("Empty matrices:", null_matrices)
print("Diagonal matrices:", diag_matrices)

#%% responders vs non-responders
response_df = pd.read_csv('../data/table_rep.csv')
old_ID = list(response_df['ID'])
new_ID = []
unknown_ID = [] # subjects for whose the connectivity matrix does not exist
for ID in response_df['ID']:
    found = False
    for patient in patients:
        if re.findall("\d+", ID)[0] == re.findall("\d+", patient)[0]:# if ID[-5:] == patient[-5:]:
            new_ID.append(patient)   
            found = True
    if not found: 
        old_ID.remove(ID)
        unknown_ID.append(ID)

response_df = response_df.replace(old_ID, new_ID)

responders = list(response_df.loc[response_df['Rep_M6_MADRS'].isin([1])]['ID'])
non_responders = list(response_df.loc[response_df['Rep_M6_MADRS'].isin([0])]['ID'])

print("\nClassified {} responders and {} non-responders".format(len(responders), len(non_responders)))

#%% Medication load
medication = pd.read_csv(csv_june_path)
old_ID = list(medication['ID'])
new_ID = []
unknown_ID = [] # subjects for whose the connectivity matrix does not exist
for ID in medication['ID']:
    found = False
    for patient in patients:
        if patient != 'lgp_164AS':
            if re.findall("\d+", ID)[0] == re.findall("\d+", patient)[0]: # substring of numbers => we only compare the ID number between the two lists
                new_ID.append(patient) 
                found = True
    if not found: 
        old_ID.remove(ID)
        unknown_ID.append(ID)
        
# remove unknown subjects
medication = medication[~medication.ID.isin(unknown_ID)]
medication = medication.replace(old_ID, new_ID)

#%% Dump data
# with open('../manage_data/data_preprocessed.pickle', 'wb') as f:
with open('../manage_data/data_glasser_preprocessed.pickle', 'wb') as f:
    pickle.dump(
        [connectivity_matrices,
          controls,
          patients,
          controls_count,
          patients_count,
          subject_count,
          patient_info_dict,
          responders, 
          non_responders,
          response_df,
          medication], f)

#%%
#!/usr/bin/env python3

# import pickle
# import random
# import copy
# import sys
# from GraphRicciCurvature.OllivierRicci import OllivierRicci
# import networkx as nx
# import bct
# import statsmodels.api as sm 

# import numpy as np
# import scipy as sp
# import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
# import itertools
# from nilearn import plotting
# from tqdm import tqdm 
# sns.set()

# # Load volumes from volumes_preprocessed.picke
# with open('../manage_data/volumes_preprocessed.pickle', 'rb') as f:
#     volumes_ROI = pickle.load(f)

# patients.sort()
# controls.sort()

# nb_ROI = len(connectivity_matrices[patients[0]])

# # TEMPORARY
# patients_to_delete = ['lgp_081LJ',
#                       'lgp_096MS',
#                       'lgp_086CA',
#                       'lgp_115LMR', # exclu
#                       'lgp_142JO', # age is NA
#                       'lgp_151MM',
#                       'lgp_164AS',
#                       ] 
# controls_to_delete = ['S166',
#                       '06001',
#                       'EMODES_003LS', # no info on excel
#                       'EMODES_004ML',
#                       ]
# subjects_to_delete = patients_to_delete + controls_to_delete

# for subject in subjects_to_delete:
#     if subject in patients:
#         patients.remove(subject)
#         patients_count -= 1
#     else:
#         print(subject)
#         controls.remove(subject)
#         controls_count -= 1

# subject_count = subject_count - len(subjects_to_delete)

# connectivity_matrices = dict([(key, val) for key, val in 
#             connectivity_matrices.items() if key not in subjects_to_delete])

# volumes_ROI = dict([(key, val) for key, val in 
#             volumes_ROI.items() if key not in subjects_to_delete])

# medication = medication[~medication.ID.isin(patients_to_delete)]

# # Density fibers connectivity matrices
# def nb_fiber2density(matrices, volumes):
#     n = list(matrices.items())[0][1].shape[0]
#     densities = copy.deepcopy(matrices)
#     densities = dict((k,v.astype(float)) for k,v in densities.items())
#     for subject, mat in densities.items():
#         for i in range(n):
#             for j in range(n):
#                 mat[i, j] = 2 * mat[i, j] / (volumes[subject][i, 1] + volumes[subject][j, 1])

#     return densities

# def get_adjacency_matrix(E, dim):
#     # vcorresp = {k:i for i,k in enumerate(np.unique(E.flatten()))}
#     # A = np.zeros((nb_ROI, nb_ROI), dtype=int)
#     # for a,b in E:
#     #     if a!=b:
#     #         A[vcorresp[a],vcorresp[b]] = A[vcorresp[b], vcorresp[a]]=1

#     A = np.zeros((dim, dim), dtype=int)
#     for i, j in E:
#         A[i, j] = 1
#     return A

# def get_parsimonious_network(matrices, ratio=0.7, ratio_fiber=0):
#     """
#     Make the matrix more parcimonious by deleting edges depending on the `ratio`
#     and the `ratio_fiber`. Also knowned as consensus threshold.

#     Parameters
#     ----------
#     connectivity_matrices : dict
#         Dictionnary of connectivity matrices.
#     ratio : float, optional
#         If the proportion of subjects >= ratio does not have an edge [i, j], then 
#         delete this edge for all subjects. 
#         The default is 0.7.
#     ratio_fiber : float, optional
#         Ratio of the fibers that connects an edge [i, j]. If None, 
#         we only see if an edge exists between i and j or not.
#         The default is 0.

#     Returns
#     -------
#     NxN ndarray.
#     Parsimonious connectivity matrix.

#     """
#     n = list(matrices.items())[0][1].shape[0]
#     threshold = ratio * len(matrices)
#     null_entries_count = np.zeros((n, n))
#     matrices_copy = copy.deepcopy(matrices)
#     key_cnt = {}

#     for mat in matrices_copy.values():
#         null_entries = np.argwhere(mat == 0)
#         for i, j in null_entries:
#             null_entries_count[i, j] += 1

#     for key, mat in matrices_copy.items():
#         # mat[null_entries_count >= threshold] = 0
#         key_cnt[key] = []
#         count = 0
#         for i in range(n):
#             for j in range(n):
#                 if null_entries_count[i, j] >= threshold and mat[i,j] != 0:
#                     mat[i, j] = 0
#                     key_cnt[key].append([i, j])
#                     count+=1
#     return key_cnt, matrices_copy

# def threshold_connections(matrices_dict, ratio=0.75):
#     matrices_copy = copy.deepcopy(matrices_dict)
#     matrices_array = np.zeros((nb_ROI, nb_ROI, subject_count))
#     i=0
#     keys = []
#     for key, mat in matrices_copy.items():
#         keys.append(key)
#         matrices_array[:, :, i] = mat
#         i+=1

#     counts = (matrices_array == 0).sum(axis=2)
#     counts = counts / subject_count

#     # ind_flat, = np.where(counts[np.triu_indices(80)] < 1-ratio)
#     ind = np.argwhere(counts > 1-ratio)
#     adj = get_adjacency_matrix(ind, nb_ROI)
#     # ixes = np.where(np.triu(np.ones((nb_ROI, nb_ROI)), 1))

#     for _, mat in matrices_copy.items():
#         mat[adj == 1] = 0

#         # mat[(ixes[0][ind_flat], ixes[1][ind_flat])] = 0
#         # mat = mat + mat.T

#     return ind, matrices_copy

# def get_network(matrix, threshold = 0):
#     """ 
#         Return the network (as a networkx data structure) defined by matrix.
#         It is possible to specify a threshold that will disregard all the 
#         edges below this threshold when creating the network
#     """
#     G = nx.Graph()
#     N = matrix.shape[0]
#     G.add_nodes_from(list(range(N)))
#     G.add_weighted_edges_from([(i,j,1.0*matrix[i][j]) for i in range(0,N) for j in range(0,i) \
#                                                                     if matrix[i][j] >= threshold])
#     return G

# def filter_weights(matrix, ratio):
#     """
#         Only keep a fraction of the weights of the matrix, fraction specified 
#         via the threshold parameter
#     """
#     n = matrix.shape[0]
#     filtered_matrix = np.zeros_like(matrix)
#     total_weight = np.sum(matrix)
#     weights_id = sorted([(matrix[i,j],i,j) for i in range(n-1) for j in range(i+1,n)], reverse=True)

#     filtered_weight = 0
#     for (w,i,j) in weights_id:
#         filtered_weight += 2*w

#         if filtered_weight > ratio*total_weight:
#             break

#         filtered_matrix[i,j] = w
#         filtered_matrix[j,i] = w

#     return filtered_matrix

# def node_curvature(matrix):
#     '''
#     Compute node curvature for each node of the matrix input.

#     Parameters
#     ----------
#     matrix : NxN np.ndarray
#         connectivity matrix.

#     Returns
#     -------
#     curvature : Nx1 np.ndarray
#         node curvature vector.

#     '''
#     n = len(matrix)
#     curvature = np.zeros((n))
#     G_nx = get_network(matrix)
#     orc = OllivierRicci(G_nx, alpha=0.5, verbose="INFO")
#     orc.compute_ricci_curvature()

#     for region_count in range(n):
#         curvature[region_count] = orc.G.nodes[region_count]['ricciCurvature']
#     return curvature

# def apply_threshold(input_, atlas, threshold):
#     '''
#     Set values which are lesser than threshold to 0.

#     Parameters
#     ----------
#     input_ : Nx1 or NxN np.ndarray
#         p values or squared matrix
#     threshold : float

#     Returns
#     -------
#     dictionary_copy : NxN np.ndarray
#         copy of the matrix where the threshold was applied

#     '''
#     atlas_copy = copy.deepcopy(atlas)

#     if len(input_.shape) == 2: # matrix NxN  
#         indices_set_to_zero = []
#         indices_set_to_zero.append([index for index in range(len(input_)) if (
#             np.unique(input_[:, index] == 0)[0])]
#             )
#         atlas_copy[indices_set_to_zero, :] = 0

#         return atlas_copy

#     else: # vector Nx1
#         indices_set_to_zero = [i for i in range(len(input_)) if input_[i] >= threshold]
#         atlas_copy[indices_set_to_zero] = 0

#         indices_set_to_one = [i for i in range(len(input_)) if input_[i] < threshold]
#         matrix = np.zeros((nb_ROI, nb_ROI))
#         for index in indices_set_to_one:
#             matrix[index][index] = 1

#         return matrix, atlas_copy

# def apply_threshold_regions(input_, atlas):
#     '''
#     Set values which are not in input_ to zero.

#     Parameters
#     ----------
#     input_ : dict
#         regions
#     atlas : np.array of shape (nb_ROI, 3)

#     Returns
#     -------


#     '''
#     atlas_copy = copy.deepcopy(atlas)

#     signif_regions = [l.tolist() for l in list(input_.values())]
#     signif_regions = list(itertools.chain(*signif_regions))

#     indices_set_to_zero = list(set(np.arange(0, nb_ROI)) - set(signif_regions))
#     atlas_copy[indices_set_to_zero] = 0

#     matrix = np.zeros((nb_ROI, nb_ROI))
#     for index in signif_regions:
#         matrix[index][index] = 1

#     return matrix, atlas_copy

# def plot_fittedvalues(y_, model):
#     plt.plot(y_, label='values')
#     plt.plot(model.fittedvalues, label='fitted values')
#     plt.axvline(x=patients_count, linestyle='--', color='red', label='Patients/Controls separation')
#     plt.ylabel('Global efficiency')
#     plt.xlabel('Subject')
#     plt.legend()
#     plt.grid(False)
#     plt.show()

# def glm_models(data_):
#     """
#     Apply Generalized Linear Model to adjust for confounds.

#     Parameters
#     ----------
#     data_ : pandas DataFrame
#         Contains columns for Intercept, confounds and the metric observed (response variable).
#         The column containing the response variable shall be named 'Metric'.

#     Returns
#     -------
#     ndarray
#         Adjusted values for the response variable.
#     """

#     glm_linear = sm.GLM.from_formula('Metric ~ Age + Gender', data_).fit()
#     # glm_linear = sm.GLM.from_formula('Metric ~ Age + Gender', data_).fit()

#     # C = data_.loc[:, data_.columns != 'Metric']
#     # beta_hat = np.linalg.inv(C.T @ C) @ C.T @ data_['Metric']
#     # return np.array(data_['Metric'] - C @ beta_hat[:, np.newaxis])

#     # return np.array(data_['Metric'] - (glm_linear.fittedvalues - np.mean(glm_linear.fittedvalues)))
#     return np.array(data_['Metric'] - glm_linear.fittedvalues)

# def plot_test(p_values, measures, p_value=0.05, test_method='mannwhitneyu', correction_method='bonferroni', n_permutations='5000', save_fig=False):
#     i=0
#     for measure in mean_measures_controls.keys():
#         plt.figure(figsize=(18, 5))
#         plt.plot(mean_measures_controls[measure], marker='o', color='darkturquoise', label='controls')
#         plt.fill_between(np.linspace(0,79,80), 
#                           mean_measures_controls[measure] - std_measures_controls[measure], 
#                           mean_measures_controls[measure] + std_measures_controls[measure],
#                           alpha=0.25,
#                           color='cyan',
#                           edgecolor='steelblue',
#                           linewidth=2)

#         plt.plot(mean_measures_patients[measure], marker='o', color='black', label='patients')
#         plt.fill_between(np.linspace(0,79,80), 
#                           mean_measures_patients[measure] - std_measures_patients[measure], 
#                           mean_measures_patients[measure] + std_measures_patients[measure],
#                           alpha=0.5,
#                           color='darkgray',
#                           edgecolor='dimgray',
#                           linewidth=2)

#         for region_count in range(nb_ROI):
#             if measure != 'charac_path' and measure != 'global_efficiency':
#                 # Bonferroni correction
#                 if correction_method=='bonferroni':
#                     if p_values_mat[measure][region_count] < p_value:
#                         plt.axvline(x=region_count, linestyle='--', color='red')
#                 elif correction_method=='fdr':
#                     # FDR correction
#                     if res_fdr_mat[measure][region_count]:
#                         plt.axvline(x=region_count, linestyle='--', color='red')
#         plt.ylabel(measures_networks[i])
#         plt.xlabel('Regions of Interest (80 ROIs)')
#         plt.title(measures_networks[i] + ' - ' + test_method + ' - ' + n_permutations + ' permutations' + ' - p value=' + str(p_value), fontweight='bold', loc='center', fontsize=16)
#         plt.xticks(np.linspace(0,79,80).astype(int), rotation=70)
#         plt.legend()
#         if save_fig:
#             plt.savefig('graph_pictures/' + test_method + '/' + n_permutations + '/' + measures_networks[i] + '.pdf')
#         plt.show()

#         fig = plt.figure(figsize=(6, 2.75))

#         matrix_map, atlas_threshold = apply_threshold(p_values_mat[measure], atlas_region_coords, p_value)

#         # remove dot at the center
#         atlas_threshold[atlas_threshold==0] = 'nan'

#         # No significative nodes
#         if len(np.unique(matrix_map)) == 1 and len(np.unique(atlas_threshold)) == 1:
#             matrix_map, atlas_threshold = np.zeros((0, 0)), np.zeros((0, 3))
#         disp = plotting.plot_connectome(matrix_map, 
#                                         atlas_threshold,
#                                         figure=fig)

#         if save_fig:
#             disp.savefig('graph_pictures/' + test_method + '/' + n_permutations + '/' + measures_networks[i] + '_brain.pdf')
#         plotting.show()
#         i+=1

# def permutation_test(list_A, list_B, mat_obs, measure, ntest=1000):
#     """
#     Perform permutation tests for graph measures. 

#     Parameters
#     ----------
#     list_A : list
#         indices or names of first group.
#     list_B : list
#         indices or names of second group.
#     mat_obs : Nx1 np.ndarray
#         observed matrix.
#     measure : string
#         name of tested measure.
#     ntest : int, optional
#         number of permutations to perform. The default is 1000.

#     Returns
#     -------
#     mat_pval : Nx1 np.ndarray
#         matrix of p-values after permutation.

#     """
#     p = mat_obs.shape[0]
#     mat_permut = np.zeros((p, ntest))
#     mat_permut_U1 = np.zeros((p, ))
#     c = int(np.floor(0.05 * ntest))

#     # 1. randomize samples
#     for t in range(ntest):
#         subset_size = len(list_A)
#         concat_subset = list_A + list_B
#         random.shuffle(concat_subset)
#         subset_A, subset_B = concat_subset[:subset_size], concat_subset[subset_size:]

#         for i in range(nb_ROI):
#             mat_permut_U1[i], _ = sp.stats.mannwhitneyu(original_measures_subjects[measure][subset_A, i], original_measures_subjects[measure][subset_B, i])
#             mat_permut_U2 = patients_count * controls_count - mat_permut_U1
#             mat_permut[:, t] = np.minimum(mat_permut_U1, mat_permut_U2)
#             # mat_permut[i, t], _, _ = sp.stats.ttest_ind(measures_subjects[measure][subset_A, i], measures_subjects[measure][subset_B, i], equal_var=False)

#     max_stat = np.max(mat_permut, axis=0)

#     # 2. Single threshold test
#     t_max = np.sort(max_stat)[::-1][c]
#     #t_max = np.percentile(np.sort(max_stat), 100 * (1 - 0.05))

#     # 3. unnormalized p-value
#     mat_pval = np.zeros((p, ))

#     for j in range(p):
#         mat_pval[j] = np.sum(mat_permut[j, :] >= mat_obs[j]) / ntest

#     return np.sort(max_stat), mat_pval

# def f_test(x_, y_):
#     f = np.var(x_, ddof=1)/np.var(y_, ddof=1)
#     dfn = x_.size - 1 
#     dfd = y_.size - 1
#     p = 1 - sp.stats.f.cdf(f, dfn, dfd)

#     return f, p

# def get_outliers(array):
#     q1 = np.quantile(array, 0.25)
#     q3 = np.quantile(array, 0.75)
#     IQR = q3 - q1
#     outliers_ = np.array(np.where(array > q3 + 2.5*IQR))
#     #outliers_150 = outliers_[(outliers_ > 150) & (outliers_ < 180)]

#     return outliers_

# #%% Conversion from fiber numbers to density and apply connection threshold
# #for patient in connectivity_matrices.keys():
# #    connectivity_matrices[patient] = filter_weights(connectivity_matrices[patient], THRESHOLD)
# #connectivity_matrices_wo_threshold = nb_fiber2density(connectivity_matrices, volumes_ROI)
# ratio=0.7
# connectivity_matrices_wo_threshold = copy.deepcopy(connectivity_matrices)
# count_parsimonious, _ = get_parsimonious_network(connectivity_matrices, ratio=ratio)
# null_regions, connectivity_matrices = threshold_connections(connectivity_matrices, ratio=ratio)
# null_regions = get_adjacency_matrix(null_regions, nb_ROI)
# connectivity_matrices = nb_fiber2density(connectivity_matrices, volumes_ROI)
# plt.imshow(null_regions)
# plt.title('ratio = {}'.format(ratio))
