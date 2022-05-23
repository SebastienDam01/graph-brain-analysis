#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os
import csv

# Scientific modules
import numpy as np
import pandas as pd

from scipy.io import loadmat

import seaborn as sns
sns.set()

# ## Configure paths for data once for all

# +

# Folder in which data is placed
data_folder = '../data/diffusion'

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
response_df = pd.read_csv('../data/Table_data.csv')
 
old_ID = list(response_df['ID'])
new_ID = []
for ID in response_df['ID']:
    for patient in patients:
        if ID[-5:] == patient[-5:]:
            new_ID.append(patient)   

response_df = response_df.replace(old_ID, new_ID)

responders = list(response_df.loc[response_df['rep'].isin([1])]['ID'])
non_responders = list(response_df.loc[response_df['rep'].isin([0])]['ID'])

print("Classified {} responders and {} non-responders".format(len(responders), len(non_responders)))

#%% Dump data

with open('../manage_data/data_preprocessed.pickle', 'wb') as f:
    pickle.dump(
        [connectivity_matrices,
          controls,
          patients,
          controls_count,
          patients_count,
          subject_count,
          patient_info_dict,
          responders, 
          non_responders], f)
