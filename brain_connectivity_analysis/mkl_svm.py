from sklearn.model_selection import ShuffleSplit
from MKLpy.metrics.pairwise import linear_kernel as lk
from MKLpy.preprocessing import kernel_normalization
from MKLpy.model_selection import train_test_split
from MKLpy.model_selection import cross_val_score
from MKLpy.algorithms import AverageMKL, EasyMKL
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.svm import SVC

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load connectivity matrices
with open('../manage_data/data_preprocessed.pickle', 'rb') as f:
    connectivity_matrices, controls, patients, controls_count, patients_count, subject_count, patient_info_dict, responders, non_responders, response_df, medication = pickle.load(f)
    
# load measure features
with open('../manage_data/features_measures.pickle', 'rb') as f:
    X_local, X_global = pickle.load(f)

# load connection features
with open('../manage_data/features_connections.pickle', 'rb') as f:
    X_connections = pickle.load(f)
    
if np.all((X_global == 0)):
    del X_global
    
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
        controls.remove(subject)
        controls_count -= 1
        
subject_count = subject_count - len(subjects_to_delete)

# Patients = 1, controls = 0    
y = np.ones((subject_count,))
y[patients_count:] = 0

K_local = lk(X_local)
K_connections = lk(X_connections)

KL = [K_local, K_connections]

KL_norm = [kernel_normalization(K) for K in KL]

mkl = AverageMKL()

# scores = cross_val_score(KL_norm, y, mkl, n_folds=10, scoring='accuracy')
# print (scores)

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
C_values = [0.001, 0.01, 1, 10, 100]
res = {}
for C in C_values:
    base_learner = SVC(C=C)
    mkl = AverageMKL(learner=base_learner)
    scores = cross_val_score(KL_norm, y, mkl, cv=cv, scoring='accuracy')
    # print (C, scores)
    res[C] = scores
    
# df = pd.DataFrame(res, columns=['C', 'Accuracy'])

#%% 
fig = plt.figure(figsize=(8, 5))
for C in res.keys():
    plt.plot(res[C], label='C = {}'.format(C))
    plt.legend()
#plt.xticks(C_values)
plt.show()

for C in res.keys():
  print(np.mean(res[C]))