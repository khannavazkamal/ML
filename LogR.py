import numpy as np
from math import exp
import pandas as pd
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
        #X[:,4] = np.power(X[:,4],2)
        #X[:,4] = (X[:,4] - (X[:,4].mean()))/(max(X[:,4])-min(X[:,4]))
        #X[:,5] = np.power(X[:,5],0.5)
        #X[:,5] = (X[:,5] - (X[:,5].mean()))/(max(X[:,5])-min(X[:,5]))
        #X[:,6] = np.power(X[:,6],3)
        #X[:,6] = (X[:,6] - (X[:,6].mean()))/(max(X[:,6])-min(X[:,6]))
        hypothesis = hypo(X,theta)
            
        log1 = np.log(hypothesis)
        log2 = np.log(1.-hypothesis)
        j = (y*log1.T) + ((1.-y)*log2.T)
        cost = sum(j)/(-1/m)
        #print(cost)

        loss = hypothesis - y

        gradient = (X.T*loss)/m
        
        theta = theta - (alpha*gradient)
        
    return theta
def get_exposure():
    FEATURES = ['Time (ms)','Channel','DigiPot Setting','Temp (C)','RH %','Voltage','Sensor Resistance (kOhms)']
    df = pd.read_csv("C:\\Users\\wasim_sajid\\Downloads\\test_files\\exposure_data.csv",names=FEATURES, header=None, dtype=float)
    df_1 = df.loc[df['Channel']==1,]
        
    DERIVED_FEATURES = ['X0','Time (ms)','Channel','Temp (C)','RH %','Voltage','Sensor Resistance (kOhms)']
    df_1['X0'] = 1.

    X = np.matrix(df_1[DERIVED_FEATURES].values,dtype=float)
    X[:,1] = (X[:,1] - (X[:,1].mean()))/(max(X[:,1])-min(X[:,1]))
    X[:,4] = (X[:,4] - (X[:,4].mean()))/(max(X[:,4])-min(X[:,4]))
    X[:,5] = (X[:,5] - (X[:,5].mean()))/(max(X[:,5])-min(X[:,5]))
    X[:,6] = (X[:,6] - (X[:,6].mean()))/(max(X[:,6])-min(X[:,6]))
    
    return X

def get_data(y_class):
    FEATURES = ['Time (ms)','Channel','DigiPot Setting','Temp (C)','RH %','Voltage','Sensor Resistance (kOhms)']
    df = pd.read_csv("C:\\Users\\wasim_sajid\\Downloads\\test_files\\sensor_data.csv",names=FEATURES, header=None, dtype=float)
    df_1 = df.loc[df['Channel']==1,]
    
    df_1['label'] = -99999
    df_1['label']= df_1['Sensor Resistance (kOhms)'].apply(lambda x: 1 if x<102.85 else 3 if x>105.5 else 2)

    #Initially normalizing using resistance diff but used another method to below
    #df_1['Resistance Diff'] = df_1['Sensor Resistance (kOhms)'] - df_1['Sensor Resistance (kOhms)'].shift(1)
    
    #df_1 = df_1.drop('Sensor Resistance (kOhms)',1)
    
    #df_1['label'] = -99999
    #df_1['label']= df_1['Resistance Diff'].apply(lambda x: 1 if x<0 else 3 if x>0 else 2)
    df_1 = df_1.replace('NaN',0)
    
    df_1['class_1'] = df_1['label'].apply(lambda x: 1 if x==1 else 0)
    df_1['class_2'] = df_1['label'].apply(lambda x: 1 if x==2 else 0)
    df_1['class_3'] = df_1['label'].apply(lambda x: 1 if x==3 else 0)
    
    DERIVED_FEATURES = ['X0','Time (ms)','Channel','Temp (C)','RH %','Voltage','Sensor Resistance (kOhms)']
    df_1['X0'] = 1.

    X = np.matrix(df_1[DERIVED_FEATURES].values,dtype=float)
    X[:,1] = (X[:,1] - (X[:,1].mean()))/(max(X[:,1])-min(X[:,1]))
    X[:,4] = (X[:,4] - (X[:,4].mean()))/(max(X[:,4])-min(X[:,4]))
    X[:,5] = (X[:,5] - (X[:,5].mean()))/(max(X[:,5])-min(X[:,5]))
    X[:,6] = (X[:,6] - (X[:,6].mean()))/(max(X[:,6])-min(X[:,6]))
    
    print('Running ...')
    if y_class == 'y_class1':
        y_class = np.matrix(df_1['class_1'].apply(lambda x: 1 if x==1 else 0)).T
        return X,y_class
    
    if y_class == 'y_class2':
        y_class = np.matrix(df_1['class_2'].apply(lambda x: 1 if x==1 else 0)).T
        return X,y_class

    if y_class == 'y_class3':
        y_class = np.matrix(df_1['class_3'].apply(lambda x: 1 if x==1 else 0)).T
        return X,y_class

# TRAINING
alpha = 0.00001
theta = np.matrix('0 1 1 1 1 1 1').T
iteration = 100
tsize = 0.9
X,y = get_data('y_class1')
X_train, X_test, y_train, y_test1 = cross_validation.train_test_split(X,y, test_size=tsize, random_state=0)
m = X_train.__len__()

new_theta_1 = np.matrix(sigmoid(theta, X_train, y_train, m, alpha,iteration))
prediction1 = hypo(X_test,new_theta_1)


X,y = get_data('y_class2')
X_train, X_test, y_train, y_test2 = cross_validation.train_test_split(X,y, test_size=tsize, random_state=0)
m = X_train.__len__()

new_theta_2 = np.matrix(sigmoid(theta, X_train, y_train, m, alpha,iteration))

prediction2 = hypo(X_test,new_theta_2)

X,y = get_data('y_class3')
X_train, X_test, y_train, y_test3 = cross_validation.train_test_split(X,y, test_size=tsize, random_state=0)
m = X_train.__len__()

new_theta_3 = np.matrix(sigmoid(theta, X_train, y_train, m, alpha,iteration))
prediction3 = hypo(X_test,new_theta_3)

# ACCURACY
final = []
for i in range(len(prediction1)):
    if prediction1[i] > max(prediction2[i],prediction3[i]):
        final.append(1)
    if prediction2[i] > max(prediction1[i],prediction3[i]):
        final.append(2)
    if prediction3[i] > max(prediction1[i],prediction2[i]):
        final.append(3)
        
correct = []        
for i in range(len(y_test1)):
    if y_test1[i] > max(y_test2[i],y_test3[i]):
        correct.append(1)
    if y_test2[i] > max(y_test1[i],y_test3[i]):
        correct.append(2)
    if y_test3[i] > max(y_test1[i],y_test2[i]):
        correct.append(3)

print('final size',len(final),'correct size', len(correct))

count = 0
for j in range(len(final)):
    if final[j]==correct[j]:
        count +=1
    #else:
        #print('got ',final[j],'exp',correct[j])
acc = float(count /len(final))        
print('Accuracy ', acc)

# TESTING ON REAL DATA
X = get_exposure()
prediction1 = hypo(X,new_theta_1)
prediction2 = hypo(X,new_theta_2)
prediction3 = hypo(X,new_theta_3)

label = [[]]
for i in range(len(prediction1)):
    if prediction1[i] > max(prediction2[i],prediction3[i]):
        label.append([1,float(prediction1[i])])
    if prediction2[i] > max(prediction1[i],prediction3[i]):
        label.append([2,float(prediction2[i])])
    if prediction3[i] > max(prediction1[i],prediction2[i]):
        label.append([3,float(prediction3[i])])

header = ['X0','Time (ms)','Channel','Temp (C)','RH %','Voltage','Sensor Resistance (kOhms)']
d1 = pd.DataFrame(X,columns=header)
d1 = d1.drop('X0',1)

d2 = pd.DataFrame(label, columns=['Label','Accuracy'])

output = pd.concat([d1,d2.shift(-1)], axis=1)       
final_file = pd.DataFrame(output)
final_file = final_file    
#final_file['Accuracy'] = acc 

final_file.to_csv("C:\\Users\\wasim_sajid\\Downloads\\test_files\\results.csv")
