'''
Created on 11-Jan-2014
dataset-har-PUC-Rio-ugulino.csv
@author: Udhayaraj Sivalingam
'''

import numpy as np
import pickle as pk
import csv
from numpy import genfromtxt
from Node import Node


class Randomtree:
    def __init__(self,max_depth):
        self.max_depth = max_depth
        
    def gini(self,classdistr):
        gini = 1 - np.sum(np.square(np.divide(classdistr,np.sum(classdistr))))
        return gini
    def classdistribution(self,data):
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
            
    def split(self,data,condition):
        splitted_data = [data[condition],data[~condition]]
        clsdistr_parent = self.classdistribution(data)
        clsdistr_child1 = self.classdistribution(splitted_data[0])
        clsdistr_child2 = self.classdistribution(splitted_data[1])
        gini_child1 = self.gini(clsdistr_child1)
        gini_child2 = self.gini(clsdistr_child2)
        gini_split = (np.sum(clsdistr_child1)/np.sum(clsdistr_parent))*gini_child1 + \
        (np.sum(clsdistr_child2)/np.sum(clsdistr_parent))*gini_child2
        return [gini_split,splitted_data[0],splitted_data[1]]
        
    def findbestsplit(self,data):
        gini_feature_selector = self.gini(self.classdistribution(data))
        child1 = []
        child2 = []
        feature_condition = ''
        feature_value = None
        feature_dimension = None
        features = np.random.choice(data.shape[1]-1,data.shape[1]-1,replace=False)
        for feature in features:
            randomdata = np.unique(data[:,feature])
            featurevalues = np.random.choice(randomdata,int(np.sqrt(randomdata.shape[0])),replace = False)
            #featurevalues = np.random.choice(randomdata,int(np.sqrt(np.sqrt(randomdata.shape[0]))),replace = False)
            #featurevalues = np.random.choice(randomdata,10,replace = False)
            #featurevalues = randomdata
            for featurevalue in featurevalues:
                [gini_split,split1,split2] = self.split(data, data[:,feature] <= featurevalue)
                if(gini_split < gini_feature_selector):
                    #print(gini_split)
                    gini_feature_selector = gini_split
                    child1 = np.copy(split1)
                    child2 = np.copy(split2)
                    feature_condition = 'less'
                    feature_dimension = feature
                    feature_value = featurevalue
                '''[gini_split,split1,split1] = self.split(data, data[:,feature] >= featurevalue)
                if(gini_split < gini_feature_selector):
                    gini_feature_selector = gini_split
                    child1 = np.copy(split1)
                    child2 = np.copy(split2) 
                    feature_condition = 'great' 
                    feature_dimension = feature
                    feature_value = featurevalue'''
        return[child1,child2,feature_dimension,feature_value,feature_condition]  
    
    def construct(self,data,node,depth):
        print('depth',depth)
        #print('inside construct',data[0])
        if self.max_depth > depth: 
            [child1,child2,feature,feature_value,feature_condition] = self.findbestsplit(data)
            node.feature = feature
            node.value = feature_value
            node.condition = feature_condition
            node.depth = depth
            node.data = data
            if len(child1) > 0 :
                print('leftdepth',depth)
                clasdistr1 = self.classdistribution(child1)
                print('leftclsdistr',clasdistr1)
                node.left = Node(clasdistr1)               
                if np.count_nonzero(clasdistr1)> 1:
                    self.construct(child1,node.left,depth+1)
                else:
                    node.left.isleaf=True 
            if len(child2) > 0 :
                #print('size2',child2.size)
                print('rightdepth',depth)
                clasdistr2 = self.classdistribution(child2)
                print('rightclsdistr',clasdistr2)
                node.right = Node(clasdistr2)
                if np.count_nonzero(clasdistr2)> 1:
                    self.construct(child2,node.right,depth+1)
                else:
                    node.right.isleaf=True 
        else:
            node.isleaf = True

    def printtree(self,root,label):
        print(label)
        root.printnode()
        if root.left is not None:
            self.printtree(root.left,'leftchild')
        if root.right is not None:
            self.printtree(root.right,'rightchild')
        # construct tree
    
    def create(self,input_data,output_object):
        print('tree side',input_data[0])
        clsdistr = self.classdistribution(input_data)
        rootnode = Node(clsdistr)
        #rootnode.data = input_data
        self.construct(input_data, rootnode, 0)
        #printtree(rootnode,'root')
        with open(output_object,'wb') as f:
            pk.dump(rootnode,f)
        with open('sub_data1_result.csv','w',newline='') as fp:
            a = csv.writer(fp,delimiter=',')
            a.writerows(rootnode.data)
#print(gini([16,9,0]))
#input_file = 'sub_data1.csv'
#output_object = '../forest/tree_feature_sqrt.pickle'
#def read_data():
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',names=True,usecols=tuple(range(19)),dtype = ['S10' for n in range(2)] + [ float for n in range(16)]+ ['S10'] )
 #   input_data = genfromtxt(input_file,delimiter = ',')
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',dtype = None )    
  #  return input_data
# Create new threads

#input_data = read_data()
#subsize = input_data.shape[0]/7330
#sub_data1 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#randomtree = Randomtree(0)
#randomtree.create(input_data, output_object)

#with open('sub_data1.csv','w',newline='') as fp:
 #   a = csv.writer(fp,delimiter=',')
  #  a.writerows(sub_data1)

