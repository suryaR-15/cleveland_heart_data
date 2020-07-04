# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:58:36 2020

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
                                    - converted to 1)
"""
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
# Split file into train and test/validation data; use 'selected_features' to
# train a classifier and score its performance on test data
# Set diagnosis of 1, 2, 3, 4, to 1 for binary classification as presence or
# absence of disease.
# =============================================================================
test_size = 0.2
diagnosis = data_file['Diagnosis']
diagnosis = (diagnosis > 0).astype(int)
data_file.drop(['Diagnosis'], axis=1, inplace=True)

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    data_file, diagnosis,
    test_size=test_size, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
score = confusion_matrix(y_test, predictions)
print('\nConfusion matrix: ')
print(score)
print('Accuracy on test set when training with all features: {0:.3f}'.format(
    (score[0][0] + score[1][1]) / len(y_test)))

feature_imp = pd.Series(classifier.coef_[0], index=data_file.columns)
feature_imp.nlargest(13).plot(kind='barh')
plt.title('Feature importances')
plt.show()
print('\nFeature importance plot available to view..')

# =============================================================================
# Use only the important features to train a new model and see performance
# When selecting features, look at absolute values of the importance metric
# since some are negatively correlated
# =============================================================================
no_features = 10
feature_index = 13 - no_features
sorted_features = feature_imp.reindex(feature_imp.abs().sort_values().index)
imp_features = sorted_features[feature_index:]
imp_features_list = imp_features.index.to_list()

classifier_2 = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(
    data_file[imp_features_list], diagnosis,
    test_size=test_size, random_state=1)
scaler_2 = StandardScaler()
X_train = scaler_2.fit_transform(X_train)
X_test = scaler_2.transform(X_test)

classifier_2.fit(X_train, y_train)
predictions = classifier_2.predict(X_test)
score = confusion_matrix(y_test, predictions)
print('Confusion matrix: ')
print(score)
print('Accuracy after using just {0} most important features: {1:.3f}'.
      format(no_features, (score[0][0] + score[1][1]) / len(y_test)))
print('The features used were: {0}'.format(imp_features_list))
