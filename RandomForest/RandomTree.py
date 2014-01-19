'''
Created on 11-Jan-2014

@author: Udhayaraj Sivalingam
'''
class RandomTree:
    class Node:
        def __init__(self,feature,condition,value,clsdistr,left,right):
            self.feature = feature
            self.conditon = condition
            self.value = value
            self.clsdistr = clsdistr
            self.left = left
            self.right = right
            
    def __init__(self,selfnode,leftnode,rightnode,depth):
        self.selfnode = selfnode
        self.leftnode = leftnode
        self.rightnode = rightnode
        self.depth = depth
        
