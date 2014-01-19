'''
Created on 16-Jan-2014

@author: sea-gull
'''
import numpy as np
from numpy import genfromtxt
import pickle as pk
from Node import Node

def read_data():
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',names=True,usecols=tuple(range(19)),dtype = ['S10' for n in range(2)] + [ float for n in range(16)]+ ['S10'] )
    input_data = genfromtxt('../data/newdataharr.csv',delimiter = ',')
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',dtype = None )    
    return input_data

def printtree(root,label):
    print(label)
    root.printnode()
    if root.left is not None:
        printtree(root.left,'leftchild')
    if root.right is not None:
        printtree(root.right,'rightchild')

def split(data,condition):
    splitted_data = [data[condition],data[~condition]]
    return [splitted_data[0],splitted_data[1]]

def validate(node,data):
    if node is not None and node.left is not None and node.right is not None and data.size > 0:
        if node.condition == 'less':
            [split1,split2] = split(data, data[:,node.feature] <= node.value )
            return np.vstack([validate(node.left,split1),validate(node.right,split2)])
        if node.condition == 'great':
            print(node.feature)
            [split1,split2] = split(data, data[:,node.feature] >= node.value)
            return np.vstack([validate(node.left,split1),validate(node.right,split2)])
    else:
        clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
        clsdistr_mat[:] = np.divide(node.clsdistr,np.sum(node.clsdistr))
        return np.append(clsdistr_mat,data,1)

def predict(node,data):
    if node is not None and node.left is not None and node.right is not None and data.size > 0:
        if node.condition == 'less':
            #[split1,split2] = split(data, data[node.feature] <= node.value )
            if data[node.feature] <= node.value:
                return predict(node.left,data)
            else:
                return predict(node.right,data)
        if node.condition == 'great':
            if data[node.feature] >= node.value:
                return predict(node.left,data)
            else:
            #print(node.feature)
            #[split1,split2] = split(data, data[node.feature] >= node.value)
                return predict(node.right,data)
    else:
        #clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
        print(node.clsdistr)
        print(node.data)
        clsdistr_mat = np.divide(node.clsdistr,np.sum(node.clsdistr))
        #print(clsdistr_mat)
        return np.append(clsdistr_mat,data,1)
    
with open('tree1.pickle','rb') as f:
    rootnode = pk.load(f)
    printtree(rootnode, 'root')
    newdata = read_data()
    shape = newdata.shape
    print(shape)
    #testdata = np.array([4.1,-0.1,2.2,1])
    #testdata = np.array([6.1,0.4,1.3])
    testdata = np.array([ 48,   65,   -1,   -6,   51,  -98,   10,  121, -121, -216,  -89, -139,
     5])
    testdata = np.array([48,65,-1,-6,51,-98,10,121,-121,-216,-89,-139,5])
    #testdata = np.array([-8,57,-192,63,8,-15,-499,-506,-613,-261,-97,-141,4])
    #testdata = np.array([-11,94,-100,-4,73,-123,13,105,-87,-163,-101,-159,3])
    print(testdata)
    prediction = predict(rootnode,testdata)
    print(prediction)
    #prediction = np.zeros(shape = (1,shape[1]+3))
    #prediction = validate(rootnode,newdata)
    #print(prediction.shape)
    #print(newdata[0])
    #print(prediction[100:120])

#printtree(rootnode,'root')