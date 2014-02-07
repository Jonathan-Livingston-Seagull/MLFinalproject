'''
Created on 24-Jan-2014

@author: sea-gull
'''
import numpy as np
import numpy.random
import matplotlib.pyplot as plt

# Generate some test data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#plt.clf()
#plt.imshow(heatmap, extent=extent)
plt.plot(x,y)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()