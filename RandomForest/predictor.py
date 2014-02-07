'''
Created on 16-Jan-2014

@author: sea-gull
'''
import numpy as np
from numpy import genfromtxt
import pickle as pk
from pylab import *
import csv

#read data for prediction(validation/Testing)
def read_data():
    #input_data = genfromtxt('../data/test_new_latest.csv',delimiter = ';')
    input_data = genfromtxt('../data/debora.csv',delimiter = ';')
    #input_data = genfromtxt('../data/Validation_Random_new.csv',delimiter = ';')
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';')
    return input_data

#split data based on condition
def split(data,condition):
    splitted_data = [data[condition],data[~condition]]
    return [splitted_data[0],splitted_data[1]]

#validate bulk data
def validate(node,data):
    if not node.isleaf:
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
    else:
        clsdistr_mat = np.zeros(shape = (data.shape[0],len(node.clsdistr)))
        clsdistr_mat[:] = np.divide(node.clsdistr,np.sum(node.clsdistr))
        return np.append(clsdistr_mat,data,1)

#predict single data
def predict(node,data):
    if not node.isleaf:
        if node.condition == 'less':
            if data[node.feature] <= node.value:
                clsdistr_mat =  predict(node.left,data)
                return clsdistr_mat
            else:
                clsdistr_mat =  predict(node.right,data)
                return clsdistr_mat
        '''if node.condition == 'great':
            if data[node.feature] < node.value:
                return predict(node.left,data)
            else:
            #print(node.feature)
            #[split1,split2] = split(data, data[node.feature] >= node.value)
                return predict(node.right,data)'''
    else:
        clsdistr_mat = np.divide(node.clsdistr,np.sum(node.clsdistr))
        return np.append(clsdistr_mat,data,1)

#create confusion matrix
def createconfmat(prediction,size):
    shape = prediction.shape 
    probability = prediction[:,0:5]
    max_label = [np.argmax(probability, axis=1)+1]
    prediction_updated =  np.hstack([np.transpose(max_label),prediction])
    with open('pred.csv','w',newline='') as fp:
        a = csv.writer(fp,delimiter=',')
        a.writerows(prediction_updated)
    confusion_matrix = np.zeros(shape=(size,size))
    for i in range(size):
        for j in range(size):
            group = np.extract((prediction_updated[:,0]==i+1) & (prediction_updated[:,shape[1]]==j+1), prediction_updated)
            confusion_matrix[i,j] = group.size
    return confusion_matrix

#visualise confusion matrix
def visconfmatrix(confusion_matrix):
    norm_conf = []
    for i in confusion_matrix:
        a = 0
        tmp_arr = []
        a = sum(i,0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    #plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
    for i, cas in enumerate(confusion_matrix):
        for j, c in enumerate(cas):
            if c>0:
                plt.text(j-.2, i+.2, c, fontsize=14)
    cb = fig.colorbar(res)
    savefig('../ConfMatrix/confmat1',format='png')
    plt.show()
    
def findaccuracy(conf_matrix):
    return (np.sum(conf_matrix.diagonal())/np.sum(conf_matrix))*100

number_of_trees = 25
prediction_array = []
for i in range(number_of_trees): 
    #with open('../forest/Trees_featuredivideby2/tree_feature_sqrt_featuredivideby2_'+str(i)+'.pickle','rb') as f:
    print('finished',i)
    #with open('../forest/report/div40data_div4feat_sqrfeatvalues/tree_dmax'+str(i)+'.pickle','rb') as f:
    with open('../forest/report/scenario_6/div15data_div4feat_sqrfeatvalues_d30_34000/tree_d30'+str(i)+'.pickle','rb') as f:

        rootnode = pk.load(f)
        input_data = read_data()
        shape = input_data.shape 
        print(shape)     
        predicted_matrix =  np.zeros(shape=(shape[0],shape[1]+5))
        for i in range(shape[0]):
            predicted_matrix[i] = predict(rootnode,input_data[i])
        prediction_shape = predicted_matrix.shape
        print(predicted_matrix.shape)
        prediction_array.append(predicted_matrix[:,[0,1,2,3,4,prediction_shape[1]-1]])
prediction_final = np.divide(np.sum(prediction_array,axis=0),number_of_trees)
#print(prediction_final)    
confusion_matrix = createconfmat(prediction_final,5)
print(confusion_matrix)
visconfmatrix(confusion_matrix)
print(findaccuracy(confusion_matrix))



        