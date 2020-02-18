#Taken from majorana_chain_example_idmrg.py
import numpy as np
import scipy.special
import copy




##################################################################################################

######################################### Plots ##############################################

import matplotlib.pyplot as plt

data=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Figs/ov_vs_time.txt")
#data=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Figs/ov_vs_chi.txt")
#data=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Figs/ov_vs_dt.txt")

ov=data[1]
dtl=data[0]

plt.xscale('log')
plt.yscale('log')
plt.xlabel("time")
plt.ylabel("1-overlap")
#plt.yticks((10**-3,10**-2,10**-1))
#plt.ylim(10**-4,10**1)
plt.scatter(dtl,ov,s=50)
#plt.savefig("/Users/jps145/TenPy2/JohnStuff/Figs/ov_vs_time.pdf")
plt.show()



