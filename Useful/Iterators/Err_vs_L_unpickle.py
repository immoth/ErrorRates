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
Ntw=200
Nt=200

##################################################################################################

######################################### Load Files ##############################################

L="vs"

import pickle

initevenfile = open("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_MPS_init_e",'rb')
psi_even=pickle.load(initevenfile)
initevenfile.close()

initoddfile = open("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_MPS_init_o",'rb')
psi_odd=pickle.load(initoddfile)
initoddfile.close()

Vmaxevenfile = open("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_MPS_Vmax_e",'rb')
psi_even_Vmax=pickle.load(Vmaxevenfile)
Vmaxevenfile.close()

Vmaxoddfile = open("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_MPS_Vmax_o",'rb')
psi_odd_Vmax=pickle.load(Vmaxoddfile)
Vmaxoddfile.close()


midevenfile = open("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_MPS_mid_e",'rb')
psi_even_mid=pickle.load(midevenfile)
midevenfile.close()

midoddfile = open("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_MPS_mid_o",'rb')
psi_odd_mid=pickle.load(midoddfile)
midoddfile.close()

evenfile = open("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_MPS_e",'rb')
psi_even_t=pickle.load(evenfile)
evenfile.close()

oddfile = open("/Users/jps145/TenPy2/JohnStuff/Useful/Data/Nt"+str(Nt)+"_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+"_MPS_o",'rb')
psi_odd_t=pickle.load(oddfile)
oddfile.close()



print("Check")
for i in range(0,len(psi_odd)):
    print("i="+str(i))
    print(psi_odd[i].overlap(psi_odd[i]))
    print(psi_odd_Vmax[i].overlap(psi_odd_Vmax[i]))
    print(psi_odd_mid[i].overlap(psi_odd_mid[i]))
    print(psi_odd_t[i].overlap(psi_odd_t[i]))
    print(psi_even[i].overlap(psi_even[i]))
    print(psi_even_Vmax[i].overlap(psi_even_Vmax[i]))
    print(psi_even_mid[i].overlap(psi_even_mid[i]))
    print(psi_even_t[i].overlap(psi_even_t[i]))


