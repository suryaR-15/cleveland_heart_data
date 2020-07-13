# cleveland_heart_data

Datafile obtained from https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Column names and entry meanings obtained from file heart-disease.names.

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
								 
Supervised learning using logistic regression 
(0, 1 - converted labels 1, 2, 3, 4 to 1) to predict presence or absence of disease

Unsupervised learning after dimensionality reduction by PCA
Classifies subjects into any one of 5 classes (0, 1, 2, 3, 4)