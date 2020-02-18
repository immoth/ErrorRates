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
"""
model_par = {
	'L': 2,			#
	'S': 0.5,
	'extra_hoppings': [ [('Sx','Sx',-4), ('Sy','Sy',8)], [('Sy','Sy',4)] ],
#	'Kz': -1.,		# Tz.Tz coupling
#	'hTx': 0,		# -h Tx
	'fracture_mpo': True,
	'conserve_Sz': False,
	'verbose': 0,			# for extra information, default verbose:1 is the best
}
"""
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

sim_par = {
	'CHI_LIST': {0:32},
	'VERBOSE': True,
	'STARTING_ENV_FROM_PSI': 2,			# explicitly breaks the symmetry in the AF phase
	'N_STEPS': 10,
	'MAX_ERROR_E' : 10**(-14),
	'MAX_ERROR_S' : 1*10**(-2),
	'MIN_STEPS' : 50, 
	'MAX_STEPS' : 3000,
	'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'e_tol': 5*10**(-15), 'tol_to_trunc': 1/5.},
}

M = mod.spin_chain_model(model_par)
np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
## Set the initial state for DMRG
initial_state = np.array( [M.up, M.up] )


print("HERE: "+str([int(M.d[i]) for i in range(0,len(M.d))]))

psi = iMPS.product_imps([int(M.d[i]) for i in range(0,len(M.d))], initial_state, dtype=float, conserve = M)

"""

psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve = M)


def run_g(g, chi_max, save_data=False):
	print "================================================================================"
	print "Running (g, chi) =", (g, chi_max)
	model_par['extra_hoppings'] = [ \
			[('Sx','Sx',-4), ('Sy','Sy',8)], \
			[('Sy','Sy',4*g)] \
		]

	## Set up the simulation
	model_par['verbose'] = 1
	M = mod.spin_chain_model(model_par)
	sim_par[ 'CHI_LIST' ] = {0:chi_max}
	print
	
	## Use the last state for DMRG
	start = time.time()
	dropped, Estat, Sstat, RP, LP = DMRG.ground_state(psi, M, sim_par)
	print "DMRG took", time.time() - start, "seconds.\n"
	start = time.time()
	sim_par['RP'] = RP
	sim_par['LP'] = LP
	
	psic = copy.deepcopy(psi)
	try:
		psic.canonical_form()
	except:
		print "CANONICALIZATION FAILED!"
		psic = psi
	print "Canonicalization took", time.time() - start, "seconds.\n"

	X = np.array([g, np.max(psic.chi)])

	S = np.average(psic.entanglement_entropy())
	E = np.average(psic.bond_expectation_value(M.H))
	xi = psic.correlation_length(num_ev = 3)

	sxsx = psic.bond_expectation_value(M.SxSx) * 4
	sysy = psic.bond_expectation_value(M.SySy) * 4
	xp_mass = sxsx[0] + sysy[1]		# Hamiltonian has - xx + yy
	SigmaX = 2 * M.Sx
	iSigmaY = (M.Sy*2j).iunary_blockwise(np.real)
	str_sy = psi.inf_correlation_function(iSigmaY, iSigmaY, sitesL=[0,1], sitesR=[0,1], OpStr=iSigmaY)
	str_sx = psi.inf_correlation_function(SigmaX, SigmaX, sitesL=[0,1], sitesR=[0,1], OpStr=SigmaX)
	print "Measurements took", time.time() - start, "seconds."

	np.set_printoptions(precision=4)
	print "Results for (g, chi) =", g, chi_max
	print "S = %s, E0 = %s, xi = %s" % (S, E, xi)
	print "Expected E = %s" % (TFI_groundstateenergy(g)/2 - 1)
	print "<sxsx> = %s, <sysy> = %s, <mass> = %s" % (sxsx, sysy, xp_mass)
	print joinstr(["<str sy> = ", str(str_sy), ",  <str sx> = ", str(str_sx)])
	packaged_measurements = np.concatenate(( [S, E], sxsx, sysy, [xp_mass, str_sx[0,1]] ))

	if save_data:
		with open(root + '_X.dat', 'a') as file:
			np.savetxt(file, np.atleast_2d( np.concatenate(( X, packaged_measurements, xi )), ) )


	#with open(pickleroot + 'c' + str(np.max(psi.chi)) + '.mps', 'w') as file:
	#	cPickle.dump(psi, file, protocol = -1)
	print



################################################################################
save_data = False
if len(sys.argv) > 1:
	save_data = True
	root = sys.argv[1]
else:
	root = './majoranachain' + str(time.time())
if save_data: print "Writing to file(s):", root


for g in np.arange(1.0, 1.01, 0.1):
	for chi in 16 * np.exp(np.arange(-3, -1) / 4.):
		run_g(g, int(chi), save_data=save_data)

print "================================================================================"
print "\nDone! (%s)" % root

"""
