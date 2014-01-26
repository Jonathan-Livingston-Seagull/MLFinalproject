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
output_object = '../forest/Trees_featuredivideby2/tree_feature_sqrt_featuredivideby2_'
number_of_trees = 25
depth_of_trees = float('inf')

def read_data():
    input_data = genfromtxt(input_file,delimiter = ';')
    return input_data

# Create new threads
input_data = read_data()
subsize = input_data.shape[0]/3
#subsize = input_data.shape[0]/7333
thread_pool = []
for i in range(number_of_trees):
    sub_data = input_data[np.random.choice(input_data.shape[0],subsize,replace=True),:]
    tree_thread = Treethread(i, "Thread-"+ str(i),depth_of_trees,output_object+str(i)+'.pickle',sub_data)
    thread_pool.append(tree_thread)
    

for tree_thread in thread_pool:
    tree_thread.start()    

'''with open('sub_data.csv','w',newline='') as fp:
    a = csv.writer(fp,delimiter=',')
    a.writerows(sub_data)'''
#sub_data2 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#treethread2 = Treethread(2, "Thread-2",20,output_object+'2.pickle',sub_data2)
#sub_data3 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#treethread3 = Treethread(3, "Thread-3",20,output_object+'3.pickle',sub_data3)
#sub_data4 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#treethread4 = Treethread(4, "Thread-4",20,output_object+'4.pickle',sub_data4)
#sub_data5 = input_data[np.random.choice(input_data.shape[0],subsize,replace=False),:]
#treethread5 = Treethread(5, "Thread-5",20,output_object+'5.pickle',sub_data5)
# Start new Threads
#treethread1.start()
#treethread2.start()
#treethread3.start()
#treethread4.start()
#treethread5.start()

print("Exiting Main Thread")