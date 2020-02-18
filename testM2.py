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

model_par = {
    'L': 4,
    'S': 0.5,
    'Jx': 1.0,
    'Jy': 1.0,
    'Jz': 0.1,
    'hz': 0.2,
    'conserve_Sz': True,  #Sz changes psi.B[].num_q to 1
    'conserce_Z2': True,  #Not sure which conserve to use.  Sz overrides Z2?
    'verbose': 0,            # for extra information, default verbose:1 is the best
}

M = mod.spin_chain_model(model_par)
np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
## Set the initial state for DMRG
initial_state = np.array( [M.up for i in range(0,model_par['L'])] )

print(["Jx",M.Jx])
print(["Jy",M.Jy])
print(["Jz",M.Jz])
print(["hz",M.hz])

print("H_mpo")
print(M.H_mpo)
print("number of conserved charges = " + str(M.H_mpo[0].num_q))
print("shape = " + str(M.H_mpo[0].shape))
print("An element in H: " + str(M.H_mpo[0].dat[0][2][2][0][0]))


## I had to convert M.d into a list of integers.  JS
## Here is the old way of doing it:  JS
## psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve = M) JS
psi = iMPS.product_imps([int(M.d[i]) for i in range(0,len(M.d))], initial_state, dtype=float, conserve = M)

print("B:")
print(psi.B)
print("number of conserved charges = " + str(psi.B[0].num_q))
print("shape = " + str(psi.B[0].shape))
print("print psi:" + str([psi.B[i].dat for i in range(0,model_par['L'])]))
print("print initial_state:" + str(initial_state))
print(psi.B[0])





print(M.H) #What is this?  Why is it different from H_mpo?






"DMRG Stuff"
#Need to understand parameters
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

v = Bunch(**{"chi_min": 128, "p": 1, "q": 3, "tL": [0.0], "Vs": {"haldane": [0.0, 0.447, 0., 0.], "Dmax":None, "eps":0.01, "TK": [0.0]}, "chi_inc": 0.2,  "svd_max": 18, "trunc_cut":10**(-8.), "LxL": [22.0], "chi_max": 256})

sim_par = {
            'VERBOSE': True,
            'N_STEPS': 2,
            'UPDATE_ENV': 0,
            'SVD_MAX' :v.svd_max,
            'TRUNC_CUT':v.trunc_cut,
            'STARTING_ENV_FROM_PSI': 0,
            'MAX_ERROR_E' :10**(-7),
            'MAX_ERROR_S' : 1*10**(-2),
            'MIN_STEPS' : 4,
            'MAX_STEPS' : 12,
            'save_mem': False,
            'LANCZOS_PAR' : {'N_min':2, 'N_max':20, 'e_tol': 10**(-12), 'tol_to_trunc': 1/10.},
}

#LP = DMRG.ground_state(psi,M,sim_par) #This is depricated apperantly
LP = DMRG.run(psi,M,sim_par)

print("JS|--------->DMRG has completed with output:")
print(LP)


"TEBD Stuff"
"Doesn't Work"

sim_par = {'chi':60,
    'VERBOSE': True,
    'TROTTER_ORDER' : 1,
    'N_STEPS' : 50,
    'MAX_ERROR_E' : 10**-12,
    'MAX_ERROR_S' : 10**-7,
    'DELTA_TAU_LIST' : [1.0*10**(-n) for n in range(4,5)],
    'DELTA_t' : 0.1
}

print(psi.L)
per_step = TEBD.time_evolution(psi,M,sim_par)


print("JS|--------->TEBD has completed with output:")
print(per_step)
