import numpy as np
from math import exp
import pandas as pd

def hypo(X,theta):
    print(X.shape)
    print(theta.shape)
    epower = -1 * np.dot(X,theta)
    print('epower',np.sum(epower, axis=0))
    e = 0 
    try:
        e = exp(np.sum(epower, axis=0))
        print('e',e)
    except Exception as e:        
        print('EXCEPTION')
        #e =np.sum(np.sum(e, axis=1).T)
        print('e',e)
        pass
    
    return np.matrix([1. / (1+e)]).T
  
def sigmoid(theta,X,y,m,alpha,iteration):
    for i in range(iteration):
        print('iteration ', i )
        #print('X', X)
        hypothesis = hypo(X,theta)
        #print('hypo',hypothesis)
            
        log1 = np.log(hypothesis)
        log2 = np.log(1.-hypothesis)
        #print('log2',log2)
        j = (y*log1.T) + ((1.-y)*log2.T)
        cost = sum(j)/(-1/m)
        #cost = np.dot(y*log1)
        print('cost',cost)
        
        loss = hypothesis - y
        #print('loss',loss)
        gradient = (X.T*loss)/m
        #print('gradient',gradient)
        
        theta = theta - (alpha*gradient)
        print('theta',theta)
        
    return theta

X_train = np.matrix('1 1 1 1 1 1 1;2 2 2 2 2 2 2;3 3 3 3 3 3 3;4 4 4 4 4 4 4;5 5 5 5 5 5 5;6 6 6 6 6 6 6;100 100 100 100 100 100 100;101 101 101 101 101 101 101;102 102 102 102 102 102 102;103 103 103 103 103 103 103;104 104 104 104 104 104 104;105 105 105 105 105 105 105;1000 1000 1000 1000 1000 1000 1000;1001 1001 1001 1001 1001 1001 1001;1002 1002 1002 1002 1002 1002 1002;1003 1003 1003 1003 1003 1003 1003;1004 1004 1004 1004 1004 1004 1004;1005 1005 1005 1005 1005 1005 1005;1006 1006 1006 1006 1006 1006 1006')
y_train_class_1 = np.matrix('1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0').T
y_train_class_2 = np.matrix('0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0').T
y_train_class_3 = np.matrix('0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1').T
X_test = np.matrix('5 6 4 6 5 7 8;105 103 105 104 107 106 100;1006 900 999 1001 1006 1005 100')
m = X_train.__len__()
alpha = 0.0005
theta = np.matrix('1 1 1 1 1 1 1').T

#print(y_train)

iteration = 10

new_theta = np.matrix(sigmoid(theta, X_train, y_train_class_1, m, alpha,iteration))
prediction1 = hypo(X_test,new_theta)
print('prediction1',prediction1)

new_theta = np.matrix(sigmoid(theta, X_train, y_train_class_2, m, alpha,iteration))
prediction2 = hypo(X_test,new_theta)
print('prediction2',prediction2)

new_theta = np.matrix(sigmoid(theta, X_train, y_train_class_3, m, alpha,iteration))
prediction3 = hypo(X_test,new_theta)
print('prediction3',prediction3)

'''
count = 0 
for i,j in prediction,y_test:
    if i==j:
        count +=1
        
print('accuracy', float(count/1308))
'''
'''
theta = np.matrix('0 1').T
X = np.matrix('1 2 5 106 108 1000 1500;1 2 3 104 107 2000 2050').T
m = X.__len__()
y = np.matrix('1 1 1 2 2 3 3').T
alpha = 0.0005

new_theta = np.matrix(sigmoid(theta, X, y, m, alpha,1))
print(new_theta)
#predict
X = np.matrix('1700 2050')
print('-'*10,'predict','-'*10)
epower = -1 * np.dot(X,new_theta)
hypothesis = np.matrix([1. / (1+(exp(i))) for i in epower]).T
print('hypo',hypothesis)
'''