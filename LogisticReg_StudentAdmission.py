# -*- coding: utf-8 -*-
"""
Problem: Implement a logistic Regression model to classify whether a student will get an admission based on score in two exams.
Training data as 3 columns (Score in Exam #1, Score in Exam #2, admitted or not)
This data set is from the machine learning course by Prof Andrew Ng, Coursera

As the training data is very small (100), the Accuracy of prediction is around 89% (low).
Higher the data size, higher the prediction accuracy. 

@author: Vishnuvardhan Janapati
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation,linear_model
 
# load data
path='your path to the data/'
df=pd.read_csv(path + 'ex2data1.txt',header=None)
df.columns=['Score_1','Score_2','Admission'] # rename columns


## Inputs (X) and labels (y) (Score #1, Score #2, and admission status)
y=np.array(df['Admission'])
X=np.array(df.drop(['Admission'],1))

Sscaler=preprocessing.StandardScaler()
Xs=Sscaler.fit_transform(X)

# Robust scaler is very helpful in handling outliers
#Rscaler=preprocessing.RobustScaler()
#Xr=Rscaler.fit_transform(X)

# logistic regression model
LogisticR=linear_model.LogisticRegression()
#
LogisticR.fit(Xs,y)
#

#print('------ Logistic Regression------------')
print('Accuracy of Linear Regression Model is ',round(LogisticR.score(Xs,y)*100,2))
#
# predicting admission status based on scores from two tests
Prediction=LogisticR.predict(Sscaler.transform(np.reshape([45.0,85.0],(1,-1))))[0]
def prediction_response(Prediction):
    if Prediction==1:
        print('Predict whether a student with [45.0 and 85.0] marks will get admission?   : Yes')
    else:
        print('Predict whether a student with [45.0 and 85.0] marks will get admission?   : No')

prediction_response(Prediction)

#s1=float(input('Please enter marks in first test  :'))
#s2=float(input('Please enter marks in second test :'))
#prediction_response(LogisticR.predict(Sscaler.transform(np.reshape([s1,s2],(1,-1))))[0])