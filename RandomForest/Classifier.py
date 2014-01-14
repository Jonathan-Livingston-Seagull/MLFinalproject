'''
Created on 11-Jan-2014
dataset-har-PUC-Rio-ugulino.csv
@author: Udhayaraj Sivalingam
'''

import numpy as np
from numpy import genfromtxt

def read_data():
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',names=True,usecols=tuple(range(19)),dtype = ['S10' for n in range(2)] + [ float for n in range(16)]+ ['S10'] )
    input_data = genfromtxt('../data/sample.csv',delimiter = ';')
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',dtype = None )    
    return input_data

def gini(classdistr):
    gini = 1 - np.sum(np.square(np.divide(classdistr,np.sum(classdistr))))
    return gini
def classdistribution(data):
    clslbl = data[:,16]
    clsdistr = [data[clslbl == 1].shape[0],data[clslbl == 2].shape[0],data[clslbl == 3].shape[0],\
    data[clslbl == 4].shape[0],data[clslbl == 5].shape[0]]
    return clsdistr
def split(data,condition):
    splitted_data = [[data[condition],data[~condition]]]
    clsdistr_parent = classdistribution(data)
    clsdistr_child1 = classdistribution(splitted_data[0])
    clsdistr_child2 = classdistribution(splitted_data[1])
    gini_child1 = gini(splitted_data[0])
    gini_child2 = gini(splitted_data[1])
    gini_split = (np.sum(clsdistr_child1)/np.sum(clsdistr_parent))*gini_child1 + \
    (np.sum(clsdistr_child2)/np.sum(clsdistr_parent))*gini_child2
    return [gini_split,splitted_data[0],splitted_data[0]]
    
def findbestsplit(data,features):
    gini_feature_selector = 1
    child1 = []
    child2 = []
    feature_condition = None
    for feature in features:
        featurevalues = np.random.choice(data[:,feature])
        for featurevalue in featurevalues:
            [gini_split,split1,split2] = split(data, data[:,feature] <= featurevalue)
            if(gini_split < gini_feature_selector):
                gini_feature_selector = gini_split
                child1 = split1
                child2 = split2
                feature_condition = 'less'
            [gini_split,child1,child2] = split(data, data[:,feature] >= featurevalue)
            if(gini_split < gini_feature_selector):
                gini_feature_selector = gini_split
                child1 = split1
                child2 = split2 
                feature_condition = 'greater'              
    return[child1,child2,feature,feature_condition]
    
    
print(gini([16,9,0]))

input_data = read_data()
classlabel = input_data[:,16]
classdistribution = classdistribution(input_data) 
print(classdistribution)
#print('shape is:',input_data.shape)
#print(input_data[1][7].dtype)
#print(input_data[:,14:])
#print(input_data[:,2:5].shape)
#print(input_data[:,0])
#A = input_data['user'][:]
#B = input_data[input_data[:,4] > 0]
#B = np.array([input_data])
#print(B.shape)
#print(B.shape)
#print(B)
#print(A.shape)
#print(np.where(str(A) is 'debora'))
#print(A[A == 'debora'])
#print([input_data['user']=='debora'])
#print(input_data['user'])