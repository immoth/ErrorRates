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



def TFI_groundstateenergy(g):
	return -(1.+g) * scipy.special.ellipe(4.*g/(1.+g)**2) * 2. / np.pi


##	Set the model parameters
##		+ sum_{even i} ( -Sx[i] Sx[i+1] + Sy[i] Sy[i+1] )
##		+ sum_{odd i}  ( Sy[i] Sy[i+1] )
##
##		The -xx (on even bonds) and yy (on odd bonds) terms gives a Majorana chain
##		The yy term (on even bonds) gaps out the free Majoranas

L=4
model_par = {
    'L': L,
    'S': 0.5,
    'Jx': 1.0,
    'Jy': 0.05,
    'Jz': 0.2,
    'hz': 0.1,
    'conserce_Z2': True,  #Not sure which conserve to use.  Sz overrides Z2?
    'verbose': 0,            # for extra information, default verbose:1 is the best
}

M = mod.spin_chain_model(model_par)
np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
## Set the initial state for DMRG
#initial_state = np.array( [M.up for i in range(0,model_par['L'])] )
initial_state_even = np.array( [1 for i in range(0,model_par['L'])] )
initial_state_odd=copy.deepcopy(initial_state_even)
initial_state_odd[0]=0


## I had to convert M.d into a list of integers.  JS
## Here is the old way of doing it:  JS
## psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve = M) JS
psi_even = iMPS.product_imps([int(M.d[i]) for i in range(0,len(M.d))], initial_state_even, dtype=float, conserve = M, bc="finite")

psi_odd = iMPS.product_imps([int(M.d[i]) for i in range(0,len(M.d))], initial_state_odd, dtype=float, conserve = M, bc="finite")


print("Even State:")
print(2*psi_even.site_expectation_value(M.Sz)) #Sz is half of sigma_z so need 2*
print("Odd State:")
print(2*psi_odd.site_expectation_value(M.Sz)) #Sz is half of sigma_z so need 2*
print("Overlap:")
print(psi_even.overlap(psi_odd))
psi0=psi_even.copy()
psi1=psi_even.copy()



"DMRG Stuff"

"Even"
chi_max = 32
sim_even = simulation.simulation(psi_even, M)
sim_even.model_par = model_par
sim_even.dmrg_par = {
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


"Odd"
chi_max = 32
sim_odd = simulation.simulation(psi_odd, M)
sim_odd.model_par = model_par
sim_odd.dmrg_par = {
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


sim_even.ground_state()
sim_odd.ground_state()


print("JS|--------->DMRG has completed")


print("Density:")
print(0.5-sim_even.psi.site_expectation_value(M.Sz))
print(0.5-sim_odd.psi.site_expectation_value(M.Sz))
print(sim_odd.psi.site_expectation_value(M.Sz)-sim_even.psi.site_expectation_value(M.Sz))
print("Energies:")
print(sim_even.sim_stats[-1]['Es'][-1])  #Is this the energy?  what's with the -1s?
print(sim_odd.sim_stats[-1]['Es'][-1])  #Is this the energy?  what's with the -1s?
quit()
print("overlaps")
#print(psi_even.overlap(sim_even.psi)) #psi has been updated along with sim.psi
print(psi0.overlap(sim_even.psi)) #in order to keep the initial psi you have to make a copy
print(psi1.overlap(sim_odd.psi))
print(sim_odd.psi.overlap(sim_even.psi))
print("Parity:")
print(psi_odd.correlation_function(M.Sz,M.Sz,[L-1],sites1=[0],OpStr=M.Sz)*2**L)
print(psi_even.correlation_function(M.Sz,M.Sz,[L-1],sites1=[0],OpStr=M.Sz)*2**L)

if 1:       # Apply Sx to site 1
    site = 1
    psi_copy = psi_odd.copy()
    psi_copy.B[site] = npc.tensordot(M.Sx[site]*2, psi_copy.B[site], axes=([1],[0]))
    print(0.5-psi_copy.site_expectation_value(M.Sz))
    print(psi_copy.correlation_function(M.Sz,M.Sz,[L-1],sites1=[0],OpStr=M.Sz)*2**L)










"TEBD Stuff"

sim_par = {'chi':30,
    'VERBOSE': True,
    'TROTTER_ORDER' : 1,
    'N_STEPS' : 50,
    'MAX_ERROR_E' : 10**-12,
    'MAX_ERROR_S' : 10**-7,
    'DELTA_TAU_LIST' : [1.0*10**(-n) for n in range(4,5)], #what does this do?
    'DELTA_t' : 0.1
}

print(psi_odd.L)
psi_odd_0=psi_odd.copy()
per_step_odd = TEBD.time_evolution(psi_odd,M,sim_par)
psi_even_0=psi_even.copy()
per_step_even = TEBD.time_evolution(psi_even,M,sim_par)

print("JS|--------->TEBD has completed")

print("overlap")
print(psi_odd.overlap(psi_odd_0))
print(abs(psi_odd.overlap(psi_odd_0))**2)
print(psi_even.overlap(psi_even_0))
print(abs(psi_even.overlap(psi_even_0))**2)



