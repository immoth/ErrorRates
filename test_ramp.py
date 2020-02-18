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

TEBD_sim = {'chi':30,
    'VERBOSE': True,
    'TROTTER_ORDER' : 2,
    'N_STEPS' : 1,  #changed from 50 to 1
    'MAX_ERROR_E' : 10**-12,
    'MAX_ERROR_S' : 10**-7,
    'DELTA_TAU_LIST' : [1.0*10**(-n) for n in range(4,5)], #what does this do?
    'DELTA_t' : 0.05
}

##################################################################################################

################################# Initialization ################################################

#initial model
L=4
hz=[0.05]*L
Jx=1.0
Jy=0.1
Jz=0.0
M0=M(hz,Jx,Jy,Jz,L)

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

"DMRG for MVmax"
Vmax=0.2
MVmax = M(hz,Jx,Jy,Vmax,L)
psi_odd_Vmax=psi_odd.copy()
psi_even_Vmax=psi_even.copy()
DMRG_sim(psi_even_Vmax,MVmax).ground_state()
DMRG_sim(psi_odd_Vmax,MVmax).ground_state()

print("JS|--------->DMRG has completed")

hz[0]*=2
hz[-1]*=2

"Ramp Up"

Nt=1000;
psi_odd_t=psi_odd.copy()
psi_even_t=psi_even.copy()
for ti in range(0,Nt):
    V=Vmax*ti/Nt
    Mt = M(hz,Jx,Jy,V,L)
    per_step_odd = TEBD.time_evolution(psi_odd_t,Mt,TEBD_sim)
    per_step_even = TEBD.time_evolution(psi_even_t,Mt,TEBD_sim)
    if(ti % 33 == 0):
        print([ti,V])

print("JS|--------->Ramp Up has completed")


print("overlap")
print(psi_odd_Vmax.overlap(psi_odd_t))
print(abs(psi_odd_Vmax.overlap(psi_odd_t))**2)
print(psi_even_Vmax.overlap(psi_even_t))
print(abs(psi_even_Vmax.overlap(psi_even_t))**2)


"""
"Wait"

Nt=50;
psi_odd_t=psi_odd.copy()
psi_even_t=psi_even.copy()
for ti in range(0,Nt):
    V=Vmax
    Mt = M(hz,Jx,Jy,V,L)
    per_step_odd = TEBD.time_evolution(psi_odd_t,Mt,TEBD_sim)
    per_step_even = TEBD.time_evolution(psi_even_t,Mt,TEBD_sim)
    if(ti % 33 == 0):
        print([ti,V])

print("JS|--------->Wait has completed")

print("overlap")
print(psi_odd_Vmax.overlap(psi_odd_t))
print(abs(psi_odd_Vmax.overlap(psi_odd_t))**2)
print(psi_even_Vmax.overlap(psi_even_t))
print(abs(psi_even_Vmax.overlap(psi_even_t))**2)


"Down"

Nt=100;
psi_odd_t=psi_odd.copy()
psi_even_t=psi_even.copy()
for ti in range(0,Nt):
    V=Vmax-Vmax*ti/Nt
    Mt = M(hz,Jx,Jy,V,L)
    per_step_odd = TEBD.time_evolution(psi_odd_t,Mt,TEBD_sim)
    per_step_even = TEBD.time_evolution(psi_even_t,Mt,TEBD_sim)
    if(ti % 33 == 0):
        print([ti,V])

print("JS|--------->Ramp Down has completed")


print("overlap")
print(psi_odd_Vmax.overlap(psi_odd_t))
print(abs(psi_odd_Vmax.overlap(psi_odd_t))**2)
print(psi_even_Vmax.overlap(psi_even_t))
print(abs(psi_even_Vmax.overlap(psi_even_t))**2)
"""
