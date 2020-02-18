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
L=25
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


print("Density:")
print(0.5-psi_even.site_expectation_value(M0.Sz))
print(0.5-psi_odd.site_expectation_value(M0.Sz))
print(psi_odd.site_expectation_value(M0.Sz)-psi_even.site_expectation_value(M0.Sz))
print("overlap")
print(psi_odd.overlap(psi_even))
print("Parity:")
print(psi_odd.correlation_function(M0.Sz,M0.Sz,[L-1],sites1=[0],OpStr=M0.Sz)*2**L)
print(psi_even.correlation_function(M0.Sz,M0.Sz,[L-1],sites1=[0],OpStr=M0.Sz)*2**L)

print("single site")
if 1:       # Apply Sx to site 1
    site = 1
    psi_copy = psi_odd.copy()
    psi_copy.B[site] = npc.tensordot(M0.Sx[site]*2, psi_copy.B[site], axes=([1],[0]))
    print(0.5-psi_copy.site_expectation_value(M0.Sz))
    print(psi_copy.correlation_function(M0.Sz,M0.Sz,[L-1],sites1=[0],OpStr=M0.Sz)*2**L)

print("single site new:")
psi_copy = psi_odd.copy()
apply_sz(psi_copy,2)
apply_sy(psi_copy,0)
apply_sx(psi_copy,4)
print(0.5-psi_copy.site_expectation_value(M0.Sz))
print(psi_copy.correlation_function(M0.Sz,M0.Sz,[L-1],sites1=[0],OpStr=M0.Sz)*2**L)


print("apply majorana operator:")
psi_copy = psi_odd.copy()
apply_gy(psi_copy,4)
apply_gx(psi_copy,1)
print(0.5-psi_copy.site_expectation_value(M0.Sz))
print(psi_copy.correlation_function(M0.Sz,M0.Sz,[L-1],sites1=[0],OpStr=M0.Sz)*2**L)

print("finding g0:")
gx0=find_gx0(psi_even,psi_odd)
gy0=find_gy0(psi_even,psi_odd)
print(gx0)
print(gy0)

print("Errors:")
print(phase_error_x(psi_odd,gx0,psi_even))
print(phase_error_y(psi_odd,gy0,psi_even))
print(parity_error(psi_even,gx0,gy0,psi_even))
print(parity_error(psi_odd,gx0,gy0,psi_odd))
