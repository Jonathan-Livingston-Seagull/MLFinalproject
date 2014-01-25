'''
Created on 16-Jan-2014

@author: sea-gull
'''
import numpy as np
from numpy import genfromtxt
import pickle as pk
from pylab import *
import csv
def read_data():
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',names=True,usecols=tuple(range(19)),dtype = ['S10' for n in range(2)] + [ float for n in range(16)]+ ['S10'] )
    input_data = genfromtxt('../data/Validation_Random.csv',delimiter = ',')
    #input_data = genfromtxt('sub_data1.csv',delimiter = ',')
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
    if node is not None and node.left is not None and node.right is not None:
        if node.condition == 'less':
            [split1,split2] = split(data, data[:,node.feature] <= node.value )
            if split1.size < 1 :
                clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
                clsdistr_mat[:] = np.divide(node.clsdistr,np.sum(node.clsdistr))
                return np.vstack([np.append(clsdistr_mat,data,1),validate(node.right,split2)])
            elif split2.size < 1:
                clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
                clsdistr_mat[:] = np.divide(node.clsdistr,np.sum(node.clsdistr))
                return np.vstack([validate(node.right,split1),np.append(clsdistr_mat,data,1)])
            else:
                return np.vstack([validate(node.left,split1),validate(node.right,split2)])
        if node.condition == 'great':
            [split1,split2] = split(data, data[:,node.feature] >= node.value)
            if split1.size < 1 :
                clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
                clsdistr_mat[:] = np.divide(node.clsdistr,np.sum(node.clsdistr))
                return np.vstack([np.append(clsdistr_mat,data,1),validate(node.right,split2)])
            elif split2.size < 1:
                clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
                clsdistr_mat[:] = np.divide(node.clsdistr,np.sum(node.clsdistr))
                return np.vstack([validate(node.right,split1),np.append(clsdistr_mat,data,1)])
            else:
                return np.vstack([validate(node.left,split1),validate(node.right,split2)])
    else:
        clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
        clsdistr_mat[:] = np.divide(node.clsdistr,np.sum(node.clsdistr))
        return np.append(clsdistr_mat,data,1)

def predict(node,data):
    #print(node.data)
    if not node.isleaf:
        #print(any(np.allclose(data,x) for x in node.data))
        #print(any((data == x).all() for x in node.data))
        #print( 3== (0== (node.data- data)).sum(1))
        #print(node.data)
        #print('data',data)
        #print('node data',node.data)
        if node.condition == 'less':
            #[split1,split2] = split(data, data[node.feature] <= node.value )
            if data[node.feature] <= node.value:
                clsdistr_mat =  predict(node.left,data)
                #print(clsdistr_mat)
                return clsdistr_mat
            else:
                clsdistr_mat =  predict(node.right,data)
                #print(clsdistr_mat)
                return clsdistr_mat
        '''if node.condition == 'great':
            if data[node.feature] < node.value:
                return predict(node.left,data)
            else:
            #print(node.feature)
            #[split1,split2] = split(data, data[node.feature] >= node.value)
                return predict(node.right,data)'''
    else:
        #clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
        #print(node.clsdistr)
        #print(node.data)
        clsdistr_mat = np.divide(node.clsdistr,np.sum(node.clsdistr))
        #print(clsdistr_mat)
        return np.append(clsdistr_mat,data,1)

def createconfmat(prediction,size):
    shape = prediction.shape 
    probability = prediction[:,0:5]
    print(shape)
    #probability = prediction
    max_label = [np.argmax(probability, axis=1)+1]
    prediction_updated =  np.hstack([np.transpose(max_label),prediction])
    with open('pred.csv','w',newline='') as fp:
        a = csv.writer(fp,delimiter=',')
        a.writerows(prediction_updated)
    confusion_matrix = np.zeros(shape=(size,size))
    for i in range(size):
        for j in range(size):
            group = np.extract((prediction_updated[:,0]==i+1) & (prediction_updated[:,shape[1]]==j+1), prediction_updated)
            #print('i is',i,'j is',j,'group is',group)
            #if i != j :
            #   print('i,j',i,j,group)
            confusion_matrix[i,j] = group.size
    return confusion_matrix

def visconfmatrix(confusion_matrix):
    norm_conf = []
    for i in confusion_matrix:
        a = 0
        tmp_arr = []
        a = sum(i,0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
    for i, cas in enumerate(confusion_matrix):
        for j, c in enumerate(cas):
            if c>0:
                plt.text(j-.2, i+.2, c, fontsize=14)
    cb = fig.colorbar(res)
    savefig('confmat1',format='png')
    plt.show()

number_of_trees = 1
prediction_array = np.zeros(number_of_trees)
for i in range(number_of_trees): 
    with open('../forest/tree_feature_sqrt1.pickle','rb') as f:
        rootnode = pk.load(f)
        #printtree(rootnode,'root')
        #printtree(rootnode, 'root')
        newdata = read_data()
        shape = newdata.shape
        #print(shape)
        #testdata = np.array([4.1,-0.1,2.2,1])
        #testdata = np.array([6.1,0.4,1.3])
        #testdata = np.array([3.0,96.0,-54.0,-19.0,23.0,-14.0,7.0,107.0,-91.0,-167.0,-95.0,-153.0,1.0])
        testdata = np.array([0.0,89.0,-45.0,-18.0,22.0,-17.0,15.0,107.0,-92.0,-163.0,-93.0,-159.0,1.0])
        prediction = predict(rootnode,testdata)
        print(prediction)
        predicted_matrix =  np.zeros(shape=(shape[0],shape[1]+5))
        for i in range(shape[0]):
            predicted_matrix[i] = predict(rootnode,newdata[i])
           
        #prediction = validate(rootnode,newdata)
        #prediction = np.array([[1,0,1],[1,0,1],[1,0,2],[1,0,2],[3,0,3],[3,0,1],[1,0,3],[2,0,3],[3,0,2]])
        print(predicted_matrix.shape)
        confusion_matrix = createconfmat(predicted_matrix,5)
        print(confusion_matrix)
        visconfmatrix(confusion_matrix)


        