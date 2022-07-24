#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:14:37 2022

@author: sdam
"""

import pickle
import copy
import os

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns
import sys

sys.path.append('../utils')
sns.set()

from utils import printProgressBar

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

volumes_ROI = dict([(key, val) for key, val in 
           volumes_ROI.items() if key not in subjects_to_delete])

# Operations on connectivity matrices
def get_laplacian(matrix):
    n = matrix.shape[0]
    return matrix.sum(axis=1) @ np.ones((n, 1)) * np.eye(n) - matrix

def get_normalized_laplacian(matrix):
    L = get_laplacian(matrix)
    sqrt_d = np.sqrt(matrix.sum(axis=1))
    return L/sqrt_d[:, None]/sqrt_d[None,:]

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

def frechet_networks(A, B):
    # as defined in Yamin 
    return np.trace(A + B - 2 * sp.linalg.sqrtm(A @ B))

def normalized_laplacian(matrix):
    # strength diagonal matrix
    D = np.diag(np.sum(matrix,axis=1))
    
    # identity
    I = np.identity(matrix.shape[0])
    
    # D^{-1/2} matrix
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    
    L = I - np.dot(D_inv_sqrt, matrix).dot(D_inv_sqrt)
    
    return L

def get_parsimonious_network(matrices, ratio=0.7, ratio_fiber=0):
    """
    Make the matrix more parcimonious by deleting edges depending on the `ratio`
    and the `ratio_fiber`. 

    Parameters
    ----------
    connectivity_matrices : dict
        Dictionnary of connectivity matrices.
    ratio : float, optional
        If the ratio of the subjects does not have an edge [i, j], then 
        delete this edge. 
        The default is 0.7.
    ratio_fiber : float, optional
        Ratio of the fibers that connects an edge [i, j]. If is None, 
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

def nb_fiber2density(matrices, volumes):
    n = list(matrices.items())[0][1].shape[0]
    densities = copy.deepcopy(matrices)
    densities = dict((k,v.astype(float)) for k,v in densities.items())
    for subject, mat in densities.items():
        for i in range(n):
            for j in range(n):
                mat[i, j] = mat[i, j] / (volumes[subject][i, 1] + volumes[subject][j, 1])
    
    return densities
#%% Get normalized laplacian for each subject
density_connectivity_matrices = nb_fiber2density(connectivity_matrices, volumes_ROI)
parsimonius_matrices = get_parsimonious_network(density_connectivity_matrices, ratio=0.85)

normalized_laplacian_matrices = {}
for subject, d in parsimonius_matrices.items():
    normalized_laplacian_matrices[subject] = normalized_laplacian(d)

#%% Compute Fréchet distance for patients and controls
def compute_frechet_networks(Lsym_matrices):
    subjects = patients + controls
    patients_to_patients = []
    patients_to_controls = []
    controls_to_controls = []
    
    printProgressBar(0, subject_count, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    # compare patients and patients
    i=0
    for patient1 in subjects:
        i+=1
        for patient2 in subjects[i:]:
            if patient1 != patient2:
                dist = frechet_networks(Lsym_matrices[patient1], 
                                        Lsym_matrices[patient2])
                if (patient1 in patients) and (patient2 in patients):
                    patients_to_patients.append(dist)
                elif ((patient1 in patients) and (patient2 in controls)) or ((patient2 in patients) and (patient1 in controls)):
                    patients_to_controls.append(dist)
                else :
                    controls_to_controls.append(dist)
                    
        printProgressBar(i, subject_count, prefix = 'Patients progress:', suffix = 'Complete', length = 50)
 
    return patients_to_patients, patients_to_controls, controls_to_controls

p2p_dist, p2c_dist, c2c_dist = compute_frechet_networks(normalized_laplacian_matrices)
    
print(len(p2p_dist) + len(p2c_dist) + len(c2c_dist))

# Cast complex values to float

c2c_dist = [float(item) for item in c2c_dist]
p2c_dist = [float(item) for item in p2c_dist]
p2p_dist = [float(item) for item in p2p_dist]
#%% Plot

plt.hist(p2c_dist, stacked=True, density=True, alpha=0.3, label='Patients')
plt.hist(c2c_dist, stacked=True, density=True, alpha=0.3, label='Controls')
sns.kdeplot(p2c_dist, shade=False, color='C0', alpha=0.3);
sns.kdeplot(c2c_dist, shade=False, color='C1', alpha=0.3);
plt.xlabel("Fréchet distance from patients to controls and controls to controls* \n *with some missing patients")
plt.legend()
#plt.savefig("brain_connectivity_analysis/graph pictures on good matrices/distance_frechet", dpi=600)

plt.show()

#%%
 
plt.figure(figsize=(4, 4))
x = ['C vs. C', 'P vs. C', 'P vs. P']
mean_distances = [np.mean(c2c_dist), np.mean(p2c_dist), np.mean(p2p_dist)]
std_distances = [np.std(c2c_dist), np.std(p2c_dist), np.std(p2p_dist)]

x_pos = [i for i, _ in enumerate((x))]

plt.bar(x_pos, mean_distances, yerr=std_distances)

plt.xticks(x_pos, x)
plt.title('Fréchet Distance')
#plt.savefig('brain_connectivity_analysis/graph pictures on good matrices/distance_frechet_bar_plot.png', dpi=600)

plt.show()
