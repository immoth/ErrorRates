import numpy as np

###########################################Parameters#############################################
L=25
hz=[0.05]*L
Jx=1.0
Jy=0.1
Jz=0.0
Vmax=0.2
dt=0.05
BD=30
Ntw=50

##################################################################################################

######################################### Plots ##############################################

import matplotlib.pyplot as plt

ph_xL=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/ph_x_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".txt",dtype='cdouble')
ph_yL=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/ph_y_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".txt",dtype='cdouble')
pr_eL=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/pr_e_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".txt",dtype='cdouble')
pr_oL=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/pr_o_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".txt",dtype='cdouble')

Ntl=[100*i for i in range(1,50)]


ph_xp=np.abs(np.imag(np.log(ph_xL)))
ph_yp=np.abs(np.imag(np.log(ph_yL)))
pr_ep=1-np.abs(pr_eL)
pr_op=1-np.abs(pr_oL)

print(np.min(ph_xp))
print(np.min(ph_yp))
print(ph_yp)

plt.figure('Phase Error x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("ramp time")
plt.ylabel("Phase Error x")
#plt.yticks((10**-3,10**-2,10**-1))
plt.ylim(np.min(ph_xp),np.max(ph_xp))
plt.scatter(Ntl,ph_xp,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Useful/Figs/ph_x_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".pdf")
plt.show()

plt.figure('Phase Error y')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("ramp time")
plt.ylabel("Phase Error y")
#plt.yticks((10**-3,10**-2,10**-1))
plt.ylim(10**-11,10**-3)
plt.scatter(Ntl,ph_yp,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Useful/Figs/ph_y_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".pdf")
plt.show()
plt.show()

plt.figure('Parity Error e')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("ramp time")
plt.ylabel("Parity Error e")
#plt.yticks((10**-3,10**-2,10**-1))
#plt.ylim(10**-11,10**1)
plt.scatter(Ntl,pr_ep,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Useful/Figs/pr_e_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".pdf")
plt.show()
plt.show()

plt.figure('Parity Error o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("ramp time")
plt.ylabel("Parity Error o")
#plt.yticks((10**-3,10**-2,10**-1))
#plt.ylim(10**-4,10**1)
plt.scatter(Ntl,pr_op,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Useful/Figs/pr_o_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".pdf")
plt.show()
plt.show()
