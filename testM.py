#Taken from majorana_chain_example_idmrg.py
import numpy as np
import scipy.special
import copy
from models import spin_chain as mod 
from mps.mps import iMPS
from algorithms import DMRG
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
    'L': 2,            #
    'S': 0.5,
    'extra_hoppings': [ [('Sx','Sx',-4), ('Sy','Sy',8)], [('Sy','Sy',4)] ],
    #    'Kz': -1.,        # Tz.Tz coupling
    #    'hTx': 0,        # -h Tx
    'fracture_mpo': True,
    'conserve_Sz': False,
    'verbose': 0,            # for extra information, default verbose:1 is the best
}



M = mod.spin_chain_model(model_par)
np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
## Set the initial state for DMRG
initial_state = np.array( [M.up, M.up] )

print(["M.extra_hoppings",M.extra_hoppings])
print(["len(M.extra_hoppings)",len(M.extra_hoppings)])
print(["Jx",M.Jx])
print(["Jy",M.Jy])
print(["Jz",M.Jz])

## I had to convert M.d into a list of integers.  JS
## Here is the old way of doing it:  JS
## psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve = M) JS
psi = iMPS.product_imps([int(M.d[i]) for i in range(0,len(M.d))], initial_state, dtype=float, conserve = M)


print(["B",psi.B])
print(["L",psi.L])
print(["Form",psi.form])
print(["bc",psi.bc])
print(["chi",psi.chi])
print(["d",psi.d])
print(["num_q",psi.num_q])
print(["mod_q",psi.mod_q])
print(["dtype",psi.dtype])

print(psi.s)

