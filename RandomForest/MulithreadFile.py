'''
Created on 30-Dec-2013

@author: sea-gull
'''
import threading

class FileWriter(threading.Thread):
    def __init__(self,fileName,content):
        threading.Thread.__init__(self)
        self.fileName = fileName
        self.content = content
    def run(self):
        file = open(self.fileName,'w')
        file.write(self.content + '\n')
        file.close
    def loop(self):
        while 1:
            print("inside loop")
            
        
thread1 = FileWriter('file1','I am first file')
thread2 = FileWriter('file2','I am second file')

thread1.start()
thread2.start()
#thread1.loop()