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

def TEBD_sim(chi,N_STEPS,DELTA_t):
    sim = {'chi':chi,
    'VERBOSE': True,
    'TROTTER_ORDER' : 1,
    'N_STEPS' : N_STEPS,
    'MAX_ERROR_E' : 10**-12, #does not seem to be used
    'MAX_ERROR_S' : 10**-7,  #does not seem to be used
    'DELTA_TAU_LIST' : [1.0*10**(-n) for n in range(4,5)], #what does this do?  #does not seem to be used
    'DELTA_t' : DELTA_t,
    }
    return sim

##################################################################################################

################################# Initialization ################################################

#initial model
L=4
hz=-0.05
Jx=-1.0
Jy=0.0
Jz=0.0
M0=M(hz,Jx,Jy,Jz,L)

## Set the initial state for DMRG
initial_state_even = np.array( [1 for i in range(0,L)] )
initial_state_odd=copy.deepcopy(initial_state_even)
initial_state_odd[0]=0
psi_even = iMPS.product_imps([int(M0.d[i]) for i in range(0,len(M0.d))], initial_state_even, dtype=float, conserve = M0, bc="finite")

psi_odd = iMPS.product_imps([int(M0.d[i]) for i in range(0,len(M0.d))], initial_state_odd, dtype=float, conserve = M0, bc="finite")


##################################################################################################

######################################### Simulations ##############################################

"DMRG"
DMRG_sim_even0=DMRG_sim(psi_even,M0)
DMRG_sim_odd0=DMRG_sim(psi_odd,M0)
DMRG_sim_even0.ground_state()
DMRG_sim_odd0.ground_state()

print("JS|--------->DMRG has completed")

ov=[]
dtl=[]
dt = 0.1
t_steps = int(5/dt)

for chi in range(1,10):
    "TEBD"
    psi_odd_0=psi_odd.copy()
    psi_even_0=psi_even.copy()
    error_odd = TEBD.time_evolution(psi_odd_0,M0,TEBD_sim(chi,t_steps,dt))
    error_even = TEBD.time_evolution(psi_even_0,M0,TEBD_sim(chi,t_steps,dt))

    print("JS|--------->TEBD has completed with chi="+str(chi))


    print("overlap")
    print(1-abs(psi_odd.overlap(psi_odd_0))**2)
    print(1-abs(psi_even.overlap(psi_even_0))**2)

    ov.append(1-abs(psi_odd.overlap(psi_odd_0))**2)
    dtl.append(chi)




##################################################################################################

######################################### Plots ##############################################

import matplotlib.pyplot as plt

np.savetxt("/Users/jps145/TenPy2/JohnStuff/Figs/ov_vs_chi.txt",np.array([dtl,ov]))

plt.xscale('log')
plt.yscale('log')
plt.xlabel("chi")
plt.ylabel("1-overlap")
plt.yticks((10**-3,10**-2,10**-1))
plt.ylim(10**-4,10**1)
plt.scatter(dtl,ov,s=50)
plt.savefig("/Users/jps145/TenPy2/JohnStuff/Figs/ov_vs_chi.pdf")
plt.show()



