'''
Created on 30-Dec-2013

@author: sea-gull
'''
#!/usr/bin/python

import threading
import numpy as np
import pickle as pk
from numpy import genfromtxt
from Node import Node
import csv
from Randomtree import Randomtree
exitFlag = 0

class Treethread (threading.Thread):
    def __init__(self,threadID,name,depth,treeobject,data):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.depth = depth
        self.treeobject = treeobject
        self.data = data
    def run(self):
        print("Constructing tree " + self.name)
        randomtree = Randomtree(self.depth)
        randomtree.create(self.data, self.treeobject)
        print ("construction over " + self.name)

input_file = '../data/input_data_pure.csv'
output_object = '../forest/tree_feature_sqrt'
def read_data():
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',names=True,usecols=tuple(range(19)),dtype = ['S10' for n in range(2)] + [ float for n in range(16)]+ ['S10'] )
    input_data = genfromtxt(input_file,delimiter = ';')
    #input_data = genfromtxt('../data/sample.csv',delimiter = ';',dtype = None )    
    return input_data
# Create new threads

input_data = read_data()
subsize = input_data.shape[0]/5
sub_data1 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
treethread1 = Treethread(1, "Thread-1",20,output_object+'1.pickle',sub_data1)
print('thread side',sub_data1[0])
with open('sub_data1.csv','w',newline='') as fp:
    a = csv.writer(fp,delimiter=',')
    a.writerows(sub_data1)
#sub_data2 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#treethread2 = Treethread(2, "Thread-2",20,output_object+'2.pickle',sub_data2)
#sub_data3 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#treethread3 = Treethread(3, "Thread-3",20,output_object+'3.pickle',sub_data3)
#sub_data4 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#treethread4 = Treethread(4, "Thread-4",20,output_object+'4.pickle',sub_data4)
#sub_data5 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#treethread5 = Treethread(5, "Thread-5",20,output_object+'5.pickle',sub_data5)
# Start new Threads
treethread1.start()
#treethread2.start()
#treethread3.start()
#treethread4.start()
#treethread5.start()

print("Exiting Main Thread")