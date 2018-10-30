# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:27:20 2018

@author: hakala24
"""

import matplotlib.pyplot as plt
import numpy as np

N = 4
GPU_means = (0.0036,0.0086,0.0160,0.0073)
GPU_95CI = (0.0009,0.0010,0.0023,0.0014)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(ind, GPU_means, width, color='g', yerr=GPU_95CI)
#ax.yaxis.grid(True, zorder=0)
ax.bar(ind, GPU_means, width, color='b', yerr=GPU_95CI, align='center', ecolor='black', alpha=0.7, capsize=5,)


# add some text for labels, title and axes ticks
ax.set_ylabel('Classification time (s)')
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind)
ax.set_xticklabels(('Small-CNN', 'VGG16', 'ResNet50', 'MobileNet'))

plt.tight_layout()
plt.savefig('GPU_barplot.pdf')
plt.show()


#ax.legend((rects1[0]), ('GPU'))
#ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
"""
def autolabel(rects):
 
    #Attach a text label above each bar displaying its height

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
"""
#autolabel(rects1)
#autolabel(rects2)
#plt.show()