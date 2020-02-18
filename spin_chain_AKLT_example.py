import numpy as np
from models import spin_chain as mod 
from mps.mps import iMPS
from algorithms import DMRG

from algorithms.linalg import np_conserved as npc
from algorithms.linalg import npc_helper

import time
import cProfile
import sys
import os


if len(sys.argv) > 1:
	# open our log file
	so = se = open(sys.argv[1], 'w', 0)
	# re-open stdout without buffering
	sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
	# redirect stdout and stderr to the log file opened above
	os.dup2(so.fileno(), sys.stdout.fileno())
	os.dup2(se.fileno(), sys.stderr.fileno())



# Set the model parameters
model_par = {
	'L': 2,			# Model parameter with unit cell of 2 (not mps unit cell!)  This can be 1 unless staggered_hz is non-zero
	'S': 1,			# Spin
	'AKLTcoupling': 1.,	# Use the AKLT model for the appropriate (integer) spin.
	                     # This sets the coefficient for the S.S term, and spin_chain model figures out the rest.
	                     # Jx,Jy,Jz should be set to zero, otherwise they will *add* to the existing parameters for the AKLT model.
	'Jz': 0.,		# Sz.Sz coupling
	'Jx': 0.,		# Sx.Sx coupling
	'Jy': 0.,		# Sy.Sy coupling
	'staggered_hz': 0.,
	'hz': 0.,		# Sz onsite term
	'hx': 0.,		# Sx onsite term breaks U(1) symmetry
	'magnetization': (0, 1),	# average magnetization, (0,1) means 0/1 = 0
	'fracture_mpo': False,
	'verbose': 2,			# for extra information, default verbose:1 is the best
	'dtype': float,		# work with real numbers (as opposed to complex)
}

sim_par = {
	'CHI_LIST': {0:12},
	'VERBOSE': True,
	'STARTING_ENV_FROM_PSI': 5,			# explicitly breaks the symmetry in the AF phase
	'N_STEPS': 10,
	'MAX_ERROR_E' : 10**(-14),
	'MAX_ERROR_S' : 1*10**(-2),
	'MIN_STEPS' : 20,
	'MAX_STEPS' : 100,
	'SVD_MAX' : 12,		## The cut can be very coarse for the AKLT states
	'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'e_tol': 5*10**(-15), 'tol_to_trunc': 1/8.},
}



np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
## Set up the model
M = mod.spin_chain_model(model_par)
## Set the initial state for DMRG
initial_state = np.array( [M.up, M.dn] )
#initial_state = np.array( [1, 1] )
psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve = M)


print "mpo:", M.H_mpo[0].sparse_stats()

def runDMRG():

	t0 = time.time()
	print "Starting iDMRG..."
	## Run the DMRG
	dropped, Estat, Sstat, RP, LP = DMRG.ground_state(psi, M, sim_par)
	print "It took", time.time() - t0, "seconds"
	t0 = time.time()
	## Put in to (right) canonical form 'B'
	psi.canonical_form(verbose = 1)
	print "correlation length:", psi.correlation_length()
	print "It took", time.time() - t0, "seconds"

#cProfile.run('runDMRG()', 'dmrgprof')
runDMRG()


ee = psi.entanglement_entropy()
diff = np.linalg.norm(ee - np.log(model_par['S'] + 1))
print "Entanglement entropy:", ee
print "Entanglement entropy - ln(S+1) (should be 0):", diff 
print

print "M.Sz", psi.site_expectation_value(M.Sz)
print

print "Entanglement spectrum:", psi.entanglement_spectrum()
print

print "Pollmann-Turner:", psi.pollmann_turner()

print "String order:"
print psi.correlation_function( M.Sz, M.Sz, sites2 = [50], OpStr = M.exp_i_pi_Sz)

