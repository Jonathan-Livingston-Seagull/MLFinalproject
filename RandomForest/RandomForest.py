'''
Created on 11-Jan-2014

@author: Udhayaraj Sivalingam
'''
class RandomTree:
    class Node:
        def __init__(self,feature,condition,value,clsdistr):
            self.feature = feature
            self.conditon = condition
            self.value = value
            self.clsdistr = clsdistr
            
    def __init__(self,selfnode,leftnode,rightnode):
        self.selfnode = selfnode
        self.leftnode = leftnode
        self.rightnode = rightnode
        
