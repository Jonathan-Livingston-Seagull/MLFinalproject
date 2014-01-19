'''
Created on 12-Jan-2014

@author: sea-gull
'''
import numpy as np


#example_array = np.array([[1 ,2, 4, 3],[3 ,5 ,4, 2],[4 ,6 ,1 ,7],[5 ,3 ,3, 1],[0, 4, 3, 2],[1, 9, 8, 3]])
#print(example_array.shape)
#print (example_array[example_array[:,0] > 2])
#A = np.array([(1 ,2, 4, 3),(3 ,5 ,4, 2),(4 ,6 ,1 ,7),(5 ,3 ,3, 1),(0, 4, 3, 2),(1, 9, 8, 3)])
#print(A.shape)
#A = np.array([[1, 2], [3, 4]])

def split(data,condition):
    splited = [data[condition],data[~condition]]
    return splited

data = np.array([[1 ,2, 4, 3],[3 ,5 ,4, 2],[4 ,6 ,1 ,7],[5 ,3 ,3, 1],[0, 4, 3, 2],[1, 9, 8, 3]])
print('origignal',data[:,0:3])
a = data[:,0:3]
b = [np.argmax(data[:,0:3], axis=1)+1]
c =  np.hstack([np.transpose(b),a])
print(c)
#splited = split(data,data[:,0]>data[:,1])
#splited[0][:,1] = 0
#splited[1][:,0] = 0
#print('split1',splited[0])
#print('split2',splited[1])