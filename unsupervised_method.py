# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:14:19 2020

@author: surya

Datafile obtained from https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Column names and entry meanings obtained from file heart-disease.names.
Missing entries in the file have been represented with '?'.

1. Age
2. Sex (Male: 1, Female: 0)
3. Chest pain type (Typical angina: 1, Atypical angina: 2,
                    Non-anginal pain: 3, Asymptomatic: 4)
4. Resting blood pressure
5. Serum cholesterol levels
6. Fasting blood sugar greater than 120 mg/dl (True: 1, False: 0)
7. Resting state ECG (Normal: 0,
                      ST-T wave abnormality: 1,
                      Probable or definite left ventricular hypertrophy: 2)
8. Maximum heart rate achieved
9. Exercise induced angina (Yes: 1, No: 0)
10. Depression in the ST wave induced by exercise relative to rest
11. Slope of the ST segment during peak exercise (Upsloping: 1,
                                                  Flat: 2,
                                                  Downsloping: 3)
12. Number of major vessels coloured by fluoroscopy (0 - 3)
13. Defects (Normal: 3, Fixed defect: 6, Reversible defect: 7)
14. Diagnosis of heart disease - angiographic disease status
                                 (< 50% diameter narrowing: 0
                                  > 50% diameter narrowing: 1, 2, 3, 4
                                 )

Unlike in supervised_method.py, here the labels 1, 2, 3, and 4 have not been
converted to 1 and a hierarchical clustering approach with 5 clusters have
been trialled.

Min. accuracy achieved is for diagnosis label 0 - 0.533
Max. accuracy achieved is for diagnosis label 5 - 0.933
"""
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# =============================================================================
# Read in the data, add column names to the dataframe and visualise the data
# to understand its relationship with the diagnosis - uncomment when required
# =============================================================================
data_path = 'processed.cleveland.data'

columns = ['Age', 'Sex', 'ChestPainType', 'BPResting', 'Cholesterol',
           'SugarFasting', 'ECGResting', 'HRMax', 'AnginaExercise',
           'ECGPeakExercise', 'ECGSlopeExercise', 'VesselsNumber',
           'Defects', 'Diagnosis']
data_file = pd.read_csv(data_path, sep=',', names=columns, na_values=['?'])
# print(data_file.head())
# print(data_file.shape)

features = data_file.columns
# for feature in features:
#     g = sns.catplot(x=feature, y='Diagnosis', data=data_file, kind='bar',
#                     height=6, palette='muted')
#     g.despine(left=True)

# =============================================================================
# Get information on missing data - very small number of entries.
# Drop those rows rather than impute.
# =============================================================================
print("Missing data in the file:")
for feature in features:
    print("{0}: {1}".format(feature, data_file[feature].isnull().sum()))
print('Dropping rows with missing entries...')
data_file.dropna(axis=0, inplace=True)

# =============================================================================
# Split file into train and test/validation data;
# train a classifier and score its performance on test data
# Map diagnosis of 1, 2, 3, 4, to 1 for binary classification as presence or
# absence of disease.
# =============================================================================
test_size = 0.1
diagnosis = data_file['Diagnosis']
data_file.drop(['Diagnosis'], axis=1, inplace=True)

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    data_file, diagnosis,
    test_size=test_size, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================
# PCA to identify the features that contribute the most to labels. The number
# of features to be selected is set automatically by option 'mle'.
# The corresponding significant feature names are displayed by identifying
# them from the largest PCA component scores.
# =============================================================================
pca = PCA(n_components='mle')
pca.fit(X_train, y_train)

n_pcs = pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
feature_names = data_file.columns
most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
feature_dict = pd.DataFrame(dic.items())
print('\nMost important features on each component, after PCA:')
print(feature_dict)

# =============================================================================
# Using only the features identified by the PCA analysis, train a model and
# evaluate its performance on the test data. The unsupervised model used is
# KMeans Clustering. Since the approach is classification, the confusion
# matrix has been calculated and accuracy estimated from it.
# =============================================================================
selected_features = feature_dict[1].to_list()
model = AgglomerativeClustering(n_clusters=5)  # 5 classes, so 5 clusters
X_train, X_test, y_train, y_test = train_test_split(
    data_file[selected_features], diagnosis,
    test_size=test_size, random_state=1)
scaler2 = StandardScaler()
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)

model.fit(X_train)
predictions = model.fit_predict(X_test)
score = multilabel_confusion_matrix(y_test, predictions)
print('\nClustering using these important features...')
print('\nTesting model...Confusion matrices for each diagnosis label: ')
print(score)
acc = []
for i in range(5):
    s = (score[i][0][0] + score[i][1][1]) / len(y_test)
    acc.append("%.3f" % s)  # format to 3 decimal places
print('Accuracy for each diagnosis label: {0}'.format(acc))
