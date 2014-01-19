'''
Created on 11-Jan-2014
dataset-har-PUC-Rio-ugulino.csv
@author: Udhayaraj Sivalingam
'''

import numpy as np
import pickle as pk
from numpy import genfromtxt
from Node import Node

max_tree_depth = 10
#input_file = '../data/test.csv'
#output_object = 'test.pickle'
input_file = '../data/input_data_pure.csv'
output_object = 'tree1.pickle'
def read_data():
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',names=True,usecols=tuple(range(19)),dtype = ['S10' for n in range(2)] + [ float for n in range(16)]+ ['S10'] )
    input_data = genfromtxt(input_file,delimiter = ';')
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',dtype = None )    
    return input_data

def gini(classdistr):
    gini = 1 - np.sum(np.square(np.divide(classdistr,np.sum(classdistr))))
    return gini
def classdistribution(data):
    clslbl = data[:,data.shape[1]-1]
    #print(clslbl)
    #print('size',data.size)
    if data.shape[0] >0:
        #clsdistr = [data[clslbl == 0].shape[0],data[clslbl == 1].shape[0],data[clslbl == 2].shape[0]]
        clsdistr = [data[clslbl == 1].shape[0],data[clslbl == 2].shape[0],data[clslbl == 3].shape[0],\
        data[clslbl == 4].shape[0],data[clslbl == 5].shape[0]]
        return clsdistr
    else:
        return []
        
def split(data,condition):
    splitted_data = [data[condition],data[~condition]]
    clsdistr_parent = classdistribution(data)
    clsdistr_child1 = classdistribution(splitted_data[0])
    clsdistr_child2 = classdistribution(splitted_data[1])
    gini_child1 = gini(clsdistr_child1)
    gini_child2 = gini(clsdistr_child2)
    gini_split = (np.sum(clsdistr_child1)/np.sum(clsdistr_parent))*gini_child1 + \
    (np.sum(clsdistr_child2)/np.sum(clsdistr_parent))*gini_child2
    return [gini_split,splitted_data[0],splitted_data[1]]
    
def findbestsplit(data):
    gini_feature_selector = gini(classdistribution(data))
    child1 = []
    child2 = []
    feature_condition = ''
    feature_value = None
    feature_dimension = None
    features = np.random.choice(data.shape[1]-1,data.shape[1]-1,replace=False)
    for feature in features:
        randomdata = np.unique(data[:,feature])
        featurevalues = np.random.choice(randomdata,int(np.sqrt(np.sqrt(randomdata.shape[0]))),replace = False)
        #featurevalues = np.random.choice(randomdata,10,replace = False)
        #featurevalues = randomdata
        for featurevalue in featurevalues:
            [gini_split,split1,split2] = split(data, data[:,feature] <= featurevalue)
            if(gini_split < gini_feature_selector):
                #print(gini_split)
                gini_feature_selector = gini_split
                child1 = np.copy(split1)
                child2 = np.copy(split2)
                feature_condition = 'less'
                feature_dimension = feature
                feature_value = featurevalue
            [gini_split,split1,split1] = split(data, data[:,feature] >= featurevalue)
            if(gini_split < gini_feature_selector):
                gini_feature_selector = gini_split
                child1 = np.copy(split1)
                child2 = np.copy(split2) 
                feature_condition = 'great' 
                feature_dimension = feature
                feature_value = featurevalue
    return[child1,child2,feature_dimension,feature_value,feature_condition]  

def createforest(data,node,depth):
    print('depth',depth)
    if max_tree_depth > depth: 
        [child1,child2,feature,feature_value,feature_condition] = findbestsplit(data)
        node.feature = feature
        node.value = feature_value
        node.condition = feature_condition
        node.depth = depth
        if child1.shape[0] > 0 :
            #print('size1',child1.size)
            clasdistr1 = classdistribution(child1)
            node.left = Node(clasdistr1)
            if np.count_nonzero(clasdistr1)>1:
                createforest(child1,node.left,depth+1)
        else:
            node.data =  data  
        if child2.shape[0] > 0 :
            #print('size2',child2.size)
            clasdistr2 = classdistribution(child2)
            node.right = Node(clasdistr2)
            if np.count_nonzero(clasdistr2)>1:
                createforest(child2,node.right,depth+1)
        else:
            node.data = data
    else:
        node.data = data
def printtree(root,label):
    print(label)
    root.printnode()
    if root.left is not None:
        printtree(root.left,'leftchild')
    if root.right is not None:
        printtree(root.right,'rightchild')
    # construct tree
    
#print(gini([16,9,0]))

input_data = read_data()
#print(input_data.shape[1])
#features = np.random.choice(16,10,replace=False)
clsdistr = classdistribution(input_data)
print(clsdistr)
rootnode = Node(clsdistr)

createforest(input_data,rootnode,0)
print(rootnode)
#printtree(rootnode,'root')
with open(output_object,'wb') as f:
    pk.dump(rootnode,f)
#print(rootnode)
#print(type(rootnode))
#createforest(input_data,)
#print(input_data[:,0])
#classlabel = input_data[:,16]
#classdistribution = classdistribution(input_data) 
#print(classdistribution)
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