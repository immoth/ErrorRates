import numpy as np
from models import spin_chain as mod 
from mps.mps import iMPS
from algorithms import TEBD
import matplotlib.pyplot as plt

from algorithms.linalg import np_conserved as npc
from algorithms.linalg import npc_helper

import time
import cProfile
import sys
import os

"""
model_par = {
	'L': 2,
	'S': 1,
	'Jz': 0.5,
	'Jx': 1.,
	'Jy': 1.,
	'hz': 0.,
	'hx': 0.,
	'staggered_hz': 0.,
	'D2': 0.,
	'D4': 0.,
	'Heisenberg2' : 0.,
	'magnetization': (0, 1),
	'fracture_mpo': False,
	'verbose': 1,
	'dtype': float,
}
"""

model_par = {
    'L': 2,
    'S': 1,
    'Jz': 0.5,
    'Jx': 1.,
    'Jy': 1.,
    'hz': 0.,
    'hx': 0.,
    'staggered_hz': 0.,
    'D2': 0.,
    'D4': 0.,
    'Heisenberg' : 0.,
    'magnetization': (0, 1),
    'fracture_mpo': False,
    'verbose': 1,
    'dtype': float,
}

sim_par = {'chi':60,
           'VERBOSE': True,
           'TROTTER_ORDER' : 1,
           'N_STEPS' : 50,
           'MAX_ERROR_E' : 10**-12,
           'MAX_ERROR_S' : 10**-7,
           'DELTA_TAU_LIST' : [1.0*10**(-n) for n in range(4,5)]}


np.set_printoptions(linewidth=2000, precision=5,threshold=4000)

M = mod.spin_chain_model(model_par)



initial_state = np.array( [M.up, M.dn] )

print("printing: " + str(M.d)

"""
psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve = M)


print "mpo:", M.H_mpo[0].sparse_stats()



def runTEBD(chi_max):
	t0 = time.time()
	sim_par['chi'] = chi_max
	print "Starting"
	per_step = TEBD.ground_state(psi,M,sim_par)
	print "It took", time.time() - t0, "seconds, with t = ", per_step , "per step. "

	#psi.canonical_form(verbose = 1)
	#print "Correlation length:", psi.correlation_length()
	print "Energy:", psi.bond_expectation_value(M.H)
	return per_step


chis = np.arange(10, 160, 20)
per_step = np.zeros( len(chis))

sim_par['DELTA_TAU_LIST'] = [1.0*10**(-n) for n in range(1,2)]
runTEBD(chis[0])


sim_par['DELTA_TAU_LIST'] = [1.0*10**(-n) for n in range(2,3)]
for i in range(len(chis)):
	per_step[i] = runTEBD(chis[i])
	chis[i] = np.max(psi.chi)

print chis
print per_step
plt.plot(chis, per_step)
#plt.show()


model_par['hx'] = 0.001
M = mod.spin_chain_model(model_par)
initial_state = np.array( [M.up, M.dn] )
psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve = M)


chis = np.arange(10, 160, 20)
per_step = np.zeros( len(chis))

sim_par['DELTA_TAU_LIST'] = [1.0*10**(-n) for n in range(1,2)]
runTEBD(chis[0])


sim_par['DELTA_TAU_LIST'] = [1.0*10**(-n) for n in range(2,3)]
for i in range(len(chis)):
	per_step[i] = runTEBD(chis[i])
	chis[i] = np.max(psi.chi)

print chis
print per_step
plt.plot(chis, per_step)
plt.show()


"""


