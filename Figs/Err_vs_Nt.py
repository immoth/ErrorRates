#Taken from majorana_chain_example_idmrg.py
import numpy as np
import scipy.special
import copy
from models import spin_chain as mod 
from mps.mps import iMPS
from algorithms import DMRG
from algorithms import TEBD
from algorithms.linalg import np_conserved as npc
from algorithms.linalg import npc_helper
from tools.string import joinstr
from algorithms import simulation

import time
import cProfile
import sys
import os


np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
def TFI_groundstateenergy(g):
	return -(1.+g) * scipy.special.ellipe(4.*g/(1.+g)**2) * 2. / np.pi

############################ Misc Functions ##############################################
def apply_sx(psi,l):
    site = l
    psi.B[site] = npc.tensordot(M0.Sx[site]*2, psi.B[site], axes=([1],[0]))

def apply_sy(psi,l):
    site = l
    psi.B[site] = npc.tensordot(M0.Sy[site]*2, psi.B[site], axes=([1],[0]))

def apply_sz(psi,l):
    site = l
    psi.B[site] = npc.tensordot(M0.Sz[site]*2, psi.B[site], axes=([1],[0]))

def apply_gx(psi,l):
    for ll in range(0,l):
        apply_sz(psi,ll)
    apply_sx(psi,l)

def apply_gy(psi,l):
    for ll in range(0,l):
        apply_sz(psi,ll)
    apply_sy(psi,l)

def find_gx0(psi_even,psi_odd):
    gx0=[]
    for l in range(0,psi_even.L):
        psi_copy=psi_even.copy()
        apply_gx(psi_copy,l)
        gx0.append(psi_copy.overlap(psi_odd))
    return gx0

def find_gy0(psi_even,psi_odd):
    gy0=[]
    for l in range(0,psi_even.L):
        psi_copy=psi_even.copy()
        apply_gy(psi_copy,l)
        gy0.append(psi_copy.overlap(psi_odd))
    return gy0

def phase_error_x(psi_1,gx0,psi_2):
    err=0
    for l in range(0,psi_1.L):
        psi_copy=psi_2.copy()
        apply_gx(psi_copy,l)
        err+=gx0[l]*psi_1.overlap(psi_copy)
    return err

def phase_error_y(psi_1,gy0,psi_2):
    err=0
    for l in range(0,psi_1.L):
        psi_copy=psi_2.copy()
        apply_gy(psi_copy,l)
        err+=gy0[l]*psi_1.overlap(psi_copy)
    return err

def parity_error(psi_1,gx0,gy0,psi_2):
    err=0
    for l in range(0,psi_1.L):
        for ll in range(0,psi_1.L):
            psi_copy=psi_2.copy()
            apply_gy(psi_copy,ll)
            apply_gx(psi_copy,l)
            err+=gx0[l]*gy0[ll]*psi_1.overlap(psi_copy)
    return err

def zap(psi,x0,xf):
    for l in range(x0,xf+1):
        apply_gx(psi,l)

########################### Model and Simulation Functions################################
def M(hz,Jx,Jy,Jz,L):
    model_par = {
        'L': L,
        'S': 0.5, #What is this?
        'Jx': Jx,
        'Jy': Jy,
        'Jz': Jz,
        'hz': hz,
        'conserce_Z2': True,
        'verbose': 0,            # for extra information, default verbose:1 is the best
    }
    mtp = mod.spin_chain_model(model_par)
    return mtp

def DMRG_sim(psi,M):
    chi_max = 32
    sim = simulation.simulation(psi, M)
    sim.model_par = {'L': M0.L,'S': 0.5, 'Jx': M0.Jx[0],'Jy': M0.Jy[0],'Jz': M0.Jz[0],'hz':M0.hz[0],'conserce_Z2': True,'verbose': 0}
    sim.dmrg_par = {
        #'CHI_LIST': {0:10, 10:20, 30:28, 60:40, 100:57, 200:80, 300:95},
        'CHI_LIST': {0:chi_max},
            'N_STEPS': 10,
            'STARTING_ENV_FROM_PSI': 1,
            'MIN_STEPS': 30,
            'MAX_STEPS': 100,
            'MAX_ERROR_E' : 1e-8,
            'MAX_ERROR_S' : 1e-8,
            'LANCZOS_PAR': {'N_min': 2, 'N_max': 20, 'p_tol': 1e-6, 'p_tol_to_trunc': 1/100., 'cache_v': np.inf},
    }
    return sim

dt=0.05
BD=30
TEBD_sim = {'chi':BD,
    'VERBOSE': True,
    'TROTTER_ORDER' : 2,
    'N_STEPS' : 1,  #changed from 50 to 1
    'MAX_ERROR_E' : 10**-12,
    'MAX_ERROR_S' : 10**-7,
    'DELTA_TAU_LIST' : [1.0*10**(-n) for n in range(4,5)], #what does this do?
    'DELTA_t' : dt
}

##################################################################################################

################################# Initialization ################################################

#initial model
L=25
hz=[0.05]*L
Jx=1.0
Jy=0.1
Jz=0.0
M0=M(hz,Jx,Jy,Jz,L)
hz2=copy.copy(hz)
hz2[0]*=2
hz2[-1]*=2

## Set the initial state for DMRG
initial_state_even = np.array( [1 for i in range(0,L)] )
initial_state_odd=copy.deepcopy(initial_state_even)
initial_state_odd[0]=0
psi_even = iMPS.product_imps([int(M0.d[i]) for i in range(0,len(M0.d))], initial_state_even, dtype=float, conserve = M0, bc="finite")

psi_odd = iMPS.product_imps([int(M0.d[i]) for i in range(0,len(M0.d))], initial_state_odd, dtype=float, conserve = M0, bc="finite")


##################################################################################################

#########################################Simulations################################################

"DMRG for M0"
DMRG_sim(psi_even,M0).ground_state()
DMRG_sim(psi_odd,M0).ground_state()

print("JS|--------->DMRG has completed")

gx0=find_gx0(psi_even,psi_odd)
gy0=find_gy0(psi_even,psi_odd)
print("JS|--------->Checking initalization of Majorana operators")
print(phase_error_x(psi_odd,gx0,psi_even))
print(phase_error_y(psi_odd,gy0,psi_even))
print(parity_error(psi_even,gx0,gy0,psi_even))
print(parity_error(psi_odd,gx0,gy0,psi_odd))

"DMRG for MVmax"
Vmax=0.2
MVmax = M(hz,Jx,Jy,Vmax,L)
psi_odd_Vmax=psi_odd.copy()
psi_even_Vmax=psi_even.copy()
DMRG_sim(psi_even_Vmax,MVmax).ground_state()
DMRG_sim(psi_odd_Vmax,MVmax).ground_state()

print("JS|--------->DMRG has completed")

ph_xl=[]
ph_yl=[]
pr_el=[]
pr_ol=[]
Ntl=[]
for Nt in range(100,5000,100):
    "Ramp Up"

    psi_odd_t=psi_odd.copy()
    psi_even_t=psi_even.copy()
    for ti in range(0,Nt):
        V=Vmax*ti/Nt
        Mt = M(hz2,Jx,Jy,V,L)
        per_step_odd = TEBD.time_evolution(psi_odd_t,Mt,TEBD_sim)
        per_step_even = TEBD.time_evolution(psi_even_t,Mt,TEBD_sim)
        if(ti % 33 == 0):
            print([ti,V])

    print("JS|--------->Ramp Up has completed")

    print("overlap after Ramp Up")
    print(1-abs(psi_odd_Vmax.overlap(psi_odd_t))**2)
    print(1-abs(psi_even_Vmax.overlap(psi_even_t))**2)

    "Zap"
    zap(psi_even_t,int(L/2),int(L/2)+1)
    zap(psi_odd_t,int(L/2),int(L/2)+1)


    "Wait"
    Ntw=50
    for ti in range(0,Ntw):
        V=Vmax
        Mt = M(hz2,Jx,Jy,V,L)
        per_step_odd = TEBD.time_evolution(psi_odd_t,Mt,TEBD_sim)
        per_step_even = TEBD.time_evolution(psi_even_t,Mt,TEBD_sim)
        if(ti % 33 == 0):
            print([ti,V])

    print("JS|--------->Wait has completed")





    "Down"
    for ti in range(0,Nt):
        V=Vmax-Vmax*ti/Nt
        Mt = M(hz2,Jx,Jy,V,L)
        per_step_odd = TEBD.time_evolution(psi_odd_t,Mt,TEBD_sim)
        per_step_even = TEBD.time_evolution(psi_even_t,Mt,TEBD_sim)
        if(ti % 33 == 0):
            print([ti,V])

    print("JS|--------->Ramp Down has completed")


    print("overlap after down")
    print(1-abs(psi_odd.overlap(psi_odd_t))**2)
    print(1-abs(psi_even.overlap(psi_even_t))**2)

    print("Final Errors")
    print(["Nt = ",Nt])
    ph_x=phase_error_x(psi_odd_t,gx0,psi_even_t)
    ph_y=phase_error_y(psi_odd_t,gy0,psi_even_t)
    pr_e=parity_error(psi_even_t,gx0,gy0,psi_even_t)
    pr_o=parity_error(psi_odd_t,gx0,gy0,psi_odd_t)
    ph_xl.append(ph_x)
    ph_yl.append(ph_y)
    pr_el.append(pr_e)
    pr_ol.append(pr_o)
    Ntl.append(Nt)
    print(ph_x)
    print(ph_y)
    print(pr_e)
    print(pr_o)


##################################################################################################

######################################### Plots ##############################################

import matplotlib.pyplot as plt

np.savetxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/ph_x_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".txt",ph_xl)
np.savetxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/ph_y_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".txt",ph_yl)
np.savetxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/pr_e_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".txt",pr_el)
np.savetxt("/Users/jps145/TenPy2/JohnStuff/Useful/Data/pr_o_Ntvs_Ntw"+str(Ntw)+"_L"+str(L)+"_Jx"+str(Jx)+"_Jy"+str(Jy)+"_hz"+str(hz[0])+"_Vmax"+str(Vmax)+"_dt"+str(dt)+"_BD"+str(BD)+".txt",pr_ol)

"""
plt.xscale('log')
plt.yscale('log')
plt.xlabel("time")
plt.ylabel("1-overlap")
#plt.yticks((10**-3,10**-2,10**-1))
#plt.ylim(10**-4,10**1)
plt.scatter(dtl,ovW,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Figs/ovW_vs_Nt_L25.pdf")
plt.show()


plt.xscale('log')
plt.yscale('log')
plt.xlabel("time")
plt.ylabel("1-overlap")
#plt.yticks((10**-3,10**-2,10**-1))
#plt.ylim(10**-4,10**1)
plt.scatter(dtl,ovD,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Figs/ovD_vs_Nt_L25.pdf")
plt.show()
"""

