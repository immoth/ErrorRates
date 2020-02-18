import numpy as np

###########################################Parameters#############################################
L=5
hz=[0.05]*L
Jx=1.0
Jy=0.1
Jz=0.0
Vmax=0.2
dt=0.05
BD=50
Ntw=1000
Nt=1000
L='vs'
##################################################################################################

######################################### Plots ##############################################

import matplotlib.pyplot as plt

ph_xL=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_ph_x.txt",dtype='cdouble')
ph_yL=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_ph_y.txt",dtype='cdouble')
pr_eL=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_pr_e.txt",dtype='cdouble')
pr_oL=np.loadtxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_pr_o.txt",dtype='cdouble')

vsl=[i for i in range(0,len(ph_xL))]


ph_xp=np.abs(np.sin(np.imag(np.log(ph_xL))))
ph_yp=np.abs(np.sin(np.imag(np.log(ph_yL))))
pr_ep=1-np.abs(pr_eL)
pr_op=1-np.abs(pr_oL)

print(np.min(ph_xp))
print(np.min(ph_yp))
print(ph_yp)
print(ph_yL)

plt.figure('Phase Error x')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel("Ramp Time")
plt.ylabel("Phase Error x")
#plt.yticks((10**-3,10**-2,10**-1))
plt.ylim(np.min(ph_xp),np.max(ph_xp))
plt.scatter(vsl,ph_xp,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Useful/Figs/ph_x_Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".pdf")
plt.show()

plt.figure('Phase Error y')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel("Ramp Time")
plt.ylabel("Phase Error y")
#plt.yticks((10**-3,10**-2,10**-1))
plt.ylim(np.min(ph_yp),np.max(ph_yp))
plt.scatter(vsl,ph_yp,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Useful/Figs/ph_y_Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".pdf")
plt.show()
plt.show()

plt.figure('Parity Error e')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel("Ramp Time")
plt.ylabel("Parity Error e")
#plt.yticks((10**-3,10**-2,10**-1))
#plt.ylim(10**-11,10**1)
plt.scatter(vsl,pr_ep,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Useful/Figs/pr_e_Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".pdf")
plt.show()
plt.show()

plt.figure('Parity Error o')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel("Ramp Time")
plt.ylabel("Parity Error o")
#plt.yticks((10**-3,10**-2,10**-1))
#plt.ylim(10**-4,10**1)
plt.scatter(vsl,pr_op,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Useful/Figs/pr_o_Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".pdf")
plt.show()
plt.show()
