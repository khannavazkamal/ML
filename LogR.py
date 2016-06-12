import numpy as np
from math import exp
import pandas as pd
import cmath
from sklearn import cross_validation 

def hypo_orig(X,theta):
    epower = -1 * np.dot(X,theta)
    e = 0 
    try:
        e = exp(np.sum(epower, axis=0))
        out = np.matrix([1. / (1+e)]).T
    except Exception as e:        
       e = exp(381)
       out = np.matrix([1. / (1+e)]).T    
    return out

def hypo(X,theta):
    x = np.dot(X,theta)
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.sum(np.sum(e, axis=1).T)  # ndim = 2 ???
  
def sigmoid(theta,X,y,m,alpha,iteration):
    for i in range(iteration):
        hypothesis = hypo(X,theta)
            
        log1 = np.log(hypothesis)
        log2 = np.log(1.-hypothesis)
        j = (y*log1.T) + ((1.-y)*log2.T)
        cost = sum(j)/(-1/m)

        loss = hypothesis - y

        gradient = (X.T*loss)/m
        
        theta = theta - (alpha*gradient)
        
    return theta

def get_data(y_class):
    FEATURES = ['Time (ms)','Channel','DigiPot Setting','Temp (C)','RH %','Voltage','Sensor Resistance (kOhms)']
    df = pd.read_csv("C:\\Users\\wasim_sajid\\Downloads\\test_files\\sensor_data.csv",names=FEATURES, header=None, dtype=float)
    df_1 = df.loc[df['Channel']==1,]
    
    df_1['Resistance Diff'] = df_1['Sensor Resistance (kOhms)'] - df_1['Sensor Resistance (kOhms)'].shift(1)
    df_1 = df_1.drop('Sensor Resistance (kOhms)',1)
    
    df_1['label'] = -99999
    df_1['label']= df_1['Resistance Diff'].apply(lambda x: 1 if x<0 else 3 if x>0 else 2)
    df_1 = df_1.replace('NaN',0)
    
    df_1['class_1'] = df_1['label'].apply(lambda x: 1 if x==1 else 0)
    df_1['class_2'] = df_1['label'].apply(lambda x: 1 if x==2 else 0)
    df_1['class_3'] = df_1['label'].apply(lambda x: 1 if x==3 else 0)
    
    DERIVED_FEATURES = ['X0','Time (ms)','Channel','Temp (C)','RH %','Voltage','Resistance Diff']
    df_1['X0'] = 1.

    X = np.matrix(df_1[DERIVED_FEATURES].values,dtype=float)
    X[:,1] = (X[:,1] - (X[:,1].mean()))/(max(X[:,1])-min(X[:,1]))
    X[:,4] = (X[:,4] - (X[:,4].mean()))/(max(X[:,4])-min(X[:,4]))
    X[:,5] = (X[:,5] - (X[:,5].mean()))/(max(X[:,5])-min(X[:,5]))
    
    if y_class == 'y_class1':
        y_class = np.matrix(df_1['class_1'].apply(lambda x: 1 if x==1 else 0)).T
        return X,y_class
    
    if y_class == 'y_class2':
        y_class = np.matrix(df_1['class_2'].apply(lambda x: 1 if x==1 else 0)).T
        return X,y_class

    if y_class == 'y_class3':
        y_class = np.matrix(df_1['class_3'].apply(lambda x: 1 if x==1 else 0)).T
        return X,y_class        

alpha = 0.0005
theta = np.matrix('0 1 1 1 1 1 1').T
iteration = 10
tsize = 0.05
X,y = get_data('y_class1')
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=tsize, random_state=0)
m = X_train.__len__()

new_theta = np.matrix(sigmoid(theta, X_train, y_train, m, alpha,iteration))
prediction1 = hypo(X_test,new_theta)
print('prediction1',prediction1)

X,y = get_data('y_class2')
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=tsize, random_state=0)
m = X_train.__len__()

new_theta = np.matrix(sigmoid(theta, X_train, y_train, m, alpha,iteration))
prediction1 = hypo(X_test,new_theta)
print('prediction2',prediction1)

X,y = get_data('y_class3')
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=tsize, random_state=0)
m = X_train.__len__()

new_theta = np.matrix(sigmoid(theta, X_train, y_train, m, alpha,iteration))
prediction1 = hypo(X_test,new_theta)
print('prediction3',prediction1)
