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
        
    #Find gini index    
    def gini(self,classdistr):
        gini = 1 - np.sum(np.square(np.divide(classdistr,np.sum(classdistr))))
        return gini
    
    #Find class distribution
    def classdistribution(self,data):
        clslbl = data[:,data.shape[1]-1]
        if data.shape[0] >0:
            #clsdistr = [data[clslbl == 0].shape[0],data[clslbl == 1].shape[0],data[clslbl == 2].shape[0]]
            clsdistr = [data[clslbl == 1].shape[0],data[clslbl == 2].shape[0],data[clslbl == 3].shape[0],\
            data[clslbl == 4].shape[0],data[clslbl == 5].shape[0]]
            return clsdistr
        else:
            return []
    
    #split the data with given condition        
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
    
    #Find best split of data(which improves gini index)    
    def findbestsplit(self,data):
        gini_feature_selector = self.gini(self.classdistribution(data))
        child1 = []
        child2 = []
        feature_condition = ''
        feature_value = None
        feature_dimension = None
        randomfeatureseed = data.shape[1]/4
        #randomfeatureseed = data.shape[1]/2
        features = np.random.choice(data.shape[1]-1,randomfeatureseed,replace=False)
        for feature in features:
            data_column_given_feature = np.unique(data[:,feature])
            #randomfeaturevaluesseed = int(np.sqrt(data_column_given_feature.shape[0]))
            randomfeaturevaluesseed = data_column_given_feature.shape[0]/4
            #randomfeaturevaluesseed = int(np.sqrt(np.sqrt(data_column_given_feature.shape[0])))
            #print('seed is',randomfeaturevaluesseed)
            featurevalues = np.random.choice(data_column_given_feature,randomfeaturevaluesseed,replace = False)
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
    
    #construct random tree
    def construct(self,data,node,depth):
        print(depth)
        if self.max_depth > depth: 
            [child1,child2,feature,feature_value,feature_condition] = self.findbestsplit(data)
            node.feature = feature
            node.value = feature_value
            node.condition = feature_condition
            node.depth = depth
            #node.data = data
            if len(child1) > 0 :
                clasdistr1 = self.classdistribution(child1)
                node.left = Node(clasdistr1)               
                if np.count_nonzero(clasdistr1)> 1:
                    self.construct(child1,node.left,depth+1)
                else:
                    node.left.isleaf=True 
            if len(child2) > 0 :
                clasdistr2 = self.classdistribution(child2)
                node.right = Node(clasdistr2)
                if np.count_nonzero(clasdistr2)> 1:
                    self.construct(child2,node.right,depth+1)
                else:
                    node.right.isleaf=True 
        else:
            node.isleaf = True

    # print tree
    def printtree(self,root,label):
        print(label)
        root.printnode()
        if root.left is not None:
            self.printtree(root.left,'leftchild')
        if root.right is not None:
            self.printtree(root.right,'rightchild')
        # construct tree
    
    # create new random tree
    def create(self,input_data,output_object):
        clsdistr = self.classdistribution(input_data)
        rootnode = Node(clsdistr)
        self.construct(input_data, rootnode, 0)
        with open(output_object,'wb') as f:
            pk.dump(rootnode,f)
        '''with open('sub_data1_result.csv','w',newline='') as fp:
            a = csv.writer(fp,delimiter=',')
            a.writerows(rootnode.data)'''

