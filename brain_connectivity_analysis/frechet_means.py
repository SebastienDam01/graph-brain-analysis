#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:29:54 2022

@author: sdam
"""

import pickle
import random
import copy
import os
from tqdm import tqdm

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

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
# patients_count = patients_count - len(subjects_to_delete)

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
                mat[i, j] = 2 * mat[i, j] / (volumes[subject][i, 1] + volumes[subject][j, 1])
    
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
    
    for mat in matrices_copy.values():
        null_entries = np.argwhere(mat == 0)
        for i, j in null_entries:
            null_entries_count[i, j] += 1
    
    for mat in matrices_copy.values():
        count = 0
        for i in range(n):
            for j in range(n):
                if null_entries_count[i, j] >= threshold and mat[i,j] != 0:
                    mat[i, j] = 0
                    count+=1
    return matrices_copy

connectivity_matrices = get_parsimonious_network(connectivity_matrices, ratio=0.85)
connectivity_matrices = nb_fiber2density(connectivity_matrices, volumes_ROI)

#%%
def apply_threshold(matrix, threshold, atlas=None):
    '''
    Set values which are lesser than threshold to 0.

    Parameters
    ----------
    matrix : NxN np.array
        connectivity matrix
    threshold : float

    Returns
    -------
    matrix_copy : NxN np.array
        copy of the matrix where the threshold was applied

    '''
    matrix_copy = copy.deepcopy(matrix)
    matrix_copy[matrix_copy <= threshold] = 0
    
    if atlas is not None:
        atlas_copy = copy.deepcopy(atlas)
        n = len(matrix)
        indices_set_to_zero = []
        indices_set_to_zero.append([index for index in range(n) if (
            np.unique(matrix_copy[:, index] == 0)[0])]
            )
        atlas_copy[indices_set_to_zero, :] = 0
        
        return matrix_copy, atlas_copy
        
    return matrix_copy

def get_laplacian(matrix):
    n = matrix.shape[0]
    return matrix.sum(axis=1) @ np.ones((n, 1)) * np.eye(n) - matrix

def frechet_mean_closed_form(L_mat, subjects):
    '''
    Closed form of the Fréchet mean for very fast computation.
    The geodesic distance used is the Log-Euclidean Riemannian Metric.

    Parameters
    ----------
    L_mat : NxN np.array
        SPD matrix
    subjects : list
        list of subjects (patients or controls)

    Returns
    -------
    NxN np.array
        Fréchet mean

    '''
    N = len(L_mat)
    log_sum = 0
    for subject in subjects:
        log_sum += sp.linalg.logm(L_mat[subject], disp=False)[0]
    return sp.linalg.expm(
        (1 / N) * log_sum
        )

def spd_permutation_test(list_A, list_B, mat_obs, ntest=1000):
    p = mat_obs.shape[0]
    mat_permut = np.zeros((p, p, ntest))
    
    # 1. randomize samples
    for t in tqdm(range(ntest)):
        subset_size = len(list_A)
        concat_subset = list_A + list_B
        random.shuffle(concat_subset)
        subset_A, subset_B = concat_subset[:subset_size], concat_subset[subset_size:]
        
        frechet_mean_subset_A = frechet_mean_closed_form(laplacian_matrices, subset_A)
        frechet_mean_subset_B = frechet_mean_closed_form(laplacian_matrices, subset_B)
        mat_permut[:, :, t] = abs(frechet_mean_subset_A - frechet_mean_subset_B)
        
    # 2. unnormalized p-value
    mat_pval = np.zeros((p, p))
    # 2.1 diagonal
    for j in range(p):
        mat_pval[j, j] = np.sum(mat_permut[j, j, :] >= mat_obs[j, j]) / ntest
    # 2.2 off-diagonal
    for i in range(p-1):
        for j in range(i+1, p):
            mat_pval[i, j] = np.sum(mat_permut[i, j, :] >= mat_obs[i, j]) / ntest
            mat_pval[j, i] = mat_pval[i, j]
            
    return mat_pval

#%% Get regularized laplacian for each subject

laplacian_matrices = {}
for subject, d in connectivity_matrices.items():
    laplacian_matrices[subject] = get_laplacian(d) + 1e-2 * np.eye(nb_ROI)
    
#%% Geometry-aware permutation testing

F = np.zeros((nb_ROI, nb_ROI))
for _ in tqdm(range(100)):
    subset_controls = random.choices(controls, k=10) # hardcoded number of controls taken for each test
    subset_patients = random.choices(patients, k=20) # hardcoded number of patients taken for each test

    frechet_mean_controls = frechet_mean_closed_form(laplacian_matrices, subset_controls)
    frechet_mean_patients = frechet_mean_closed_form(laplacian_matrices, subset_patients)
    frechet_mean_obs = abs(frechet_mean_controls - frechet_mean_patients)
    
    p_values_mat = spd_permutation_test(subset_controls,
                                        subset_patients,
                                        frechet_mean_obs,
                                        100)
    
    for i in range(nb_ROI):
        for j in range(nb_ROI):
            if p_values_mat[i, j] < 0.001:
                F[i, j] += 1
                
np.fill_diagonal(F, 0)
#%%
plt.imshow(F, cmap='YlOrBr')
plt.grid(False)
plt.colorbar(label="Nombre de connexions significatives")
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.savefig('brain_connectivity_analysis/graph_pictures/frechet_means.pdf')
plt.show()

#%% Brain visualization
from nilearn import plotting

fig = plt.figure(figsize=(6, 2.75))
atlas_region_coords = np.loadtxt('data/COG_free_80s.txt')
F_threshold, atlas_threshold = apply_threshold(F, 10, atlas_region_coords)
disp = plotting.plot_connectome(F_threshold, 
                                atlas_threshold,
                                figure=fig)
disp.savefig('brain_connectivity_analysis/graph_pictures/frechet_mean_brain.pdf')
plotting.show()