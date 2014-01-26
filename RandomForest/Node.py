'''
Created on 15-Jan-2014

@author: sea-gull
'''
class Node:
    #feature,conditon,featurevalue,clsdistr,left,right = 0,None,0,None,None,None
    def __init__(self,clsdistr):
        self.depth = None
        self.feature = None
        self.condition = ''
        self.value = None
        self.clsdistr = clsdistr
        self.left = None
        self.right = None
        #self.data = None
        self.isleaf = False
        
    def printnode(self):
        print('Depth',self.depth,'feature',self.feature,'condition',self.condition,'featurevalue',self.value,'classdistr',self.clsdistr,'isleaf',self.isleaf)
        
    def __str__(self, depth=0):
        ret = ""
    # Print right branch
        if self.right != None:
            ret += self.right.__str__(depth + 1)
    # Print own value
        ret += "\n" + ("    "*depth) + str(self.clsdistr)
    # Print left branch
        if self.left != None:
            ret += self.left.__str__(depth + 1)
        return ret
        
            
            