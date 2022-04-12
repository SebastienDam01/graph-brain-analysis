#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os

# Scientific modules
import numpy as np

import seaborn as sns
sns.set()

os.chdir('..')

# ## Configure paths for data once for all

# +

# Folder in which data is placed
data_folder = 'data/volumes'

#%% Rename old prefix by new prefix of volumes if needed
'''
def renamePrefix(data_folder, old_prefix, new_prefix):
    for filename in os.listdir(data_folder):
        if filename.startswith(old_prefix):
            print(filename)
            os.rename(os.path.join(data_folder, filename), os.path.join(data_folder, filename.replace(old_prefix, new_prefix)))
            
        if len(filename) > 18 and filename[18] == '_':
            print(filename, filename[:18] + filename[19:])
            os.rename(os.path.join(data_folder, filename), os.path.join(data_folder, filename[:18] + filename[19:]))
            
renamePrefix(data_folder, 'volume_ROI_dwi_', 'volume_ROI_')
'''
#%%

# prefix at the beginning of each .txt file; it is specified here in order to select 
# relevant files, as well as to make file name lighter during loading for 
# further operations (such as printing subjects names), since it does not carry 
# additional information.
prefix = 'volume_ROI_'

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
# csv_path = data_folder + "/table.csv"


# -

# ## Load data

# +
def get_matrix_file_list(data_folder, prefix):
    """ 
        Return the list of files in the folder data_folder beginning by prefix
    """
    file_list = [f for f in os.listdir(data_folder) if f.startswith(prefix)]
    return list(map(lambda x: data_folder + '/' + x, file_list))

# Create a dictionnary of all the matrices, where each matrix gets associated to 
# the filename of the corresponding .mat file minus the suffix.
volumes_ROI = {}

for f in get_matrix_file_list(data_folder, prefix):
    if os.path.getsize(f) != 0: # prevent from loading empty files
        volumes_ROI[f.replace(prefix,'').replace(data_folder + '/','').replace('.txt','')] = np.loadtxt(f)
    
# Create a dictionnary of metadata for each patient, obtained from the file 
# at csv_path (by default 'data_foler/table.csv')
patient_info_dict = {}

'''
with open(csv_path, 'r') as csv_file:
    metadata = csv.DictReader(csv_file)
    # Each patient is associated to a dictionnary containing all its information
    for row in metadata:
        metadata_dict = {key:row[key] for key in metadata.fieldnames if key != 'Subject'}
        patient_info_dict[row['Subject']] = metadata_dict
'''
     
print("Succesfully loaded {} matrices from {}.".format(len(volumes_ROI), data_folder))
#print("Metadata has been found in {} for {} subjects.".format(csv_path,len(patient_info_dict)))
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

for key in [*volumes_ROI]:
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
#%%
with open('manage_data/volumes_preprocessed.pickle', 'wb') as f:
    pickle.dump(volumes_ROI, f)