import time
import string
import sys
import os
import copy
import numpy as np
import cPickle as pickle
import cProfile
import gzip
import json
from mps.mps import iMPS
from algorithms import DMRG
from algorithms.linalg import np_conserved as npc
from algorithms.linalg import npc_helper as npc_helper
from models import cold_hof_model as mod
from tools.string import to_mathematica_lists
from tools.string import uni_to_str
from scipy.integrate import quad as integr
import fractions
from cluster import omp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-L',help='use like this: -L 2',type=int)
parser.add_argument('-U',help='use like this: -U 6',type=int)
parser.add_argument('-delta',help='use like this: -V 6',type=int)
parser.add_argument('-chi',help='use like this: -chi 200',type=int)


args = parser.parse_args()


def run(): 
	model_par  = {
		                'conserve particle number': True,
				'L': L,
                                'N': 2,
                                'Jx': 1.,
				'Jy': 1.,
                                'U': 0.,
                                'kappa': 0.,
                                'phi_0': 0.,
                                'phi_ext': 0.,
                                'sg':1,
                                'delta': 0.,
				'data_type': complex
			       }

	sim_par = {
				'CHI_LIST': {0:chi},
				'VERBOSE': True,
				'STARTING_ENV_FROM_PSI': 1,
				'N_STEPS': 5,
				'MAX_ERROR_E' : 10**(-8),
				'MAX_ERROR_S' : 1*10**(-2),
				#'MIN_STEPS' : 40,
				'TRUNC_CUT': 10**(-8),
				'MAX_STEPS' : 2400,
				'LANCZOS_PAR' : {'N_min': 2, 'N_max': 40, 'e_tol': 5*10**(-15), 'tol_to_trunc': 1/5.}
			       }

	M = mod.cold_hof_model(model_par)

        if L == 8:
                initial_state = np.array([0,1,0,1,0,0,1,0, 1,0,0,1,0,0,0,0, 0,1,0,0,0,1,0,0, 1,0,0,0,0,0,0,0])

        psi = iMPS.product_imps(M.d, initial_state, dtype=complex, conserve = M)
	
        print 'Maximal number of bosons per site:', M.num
        print 'L =', M.circ
        print 'Jx =', M.Jx
        print 'Jy =', M.Jy
        print 'U =', M.U
        print 'delta =', M.delta
        print 'phi_ext =', M.phi_ext
        print 'kappa =', M.kappa
        print 'omega =', M.omega
        print 'phi_0 =', M.phi_0

        print 'initial state =', initial_state

	start = time.time()

	out = DMRG.run(psi,M,sim_par)

	zeit = time.time() - start

	print 'DMRG took', zeit,'seconds.'      


        print 'Computing momentum...'
        U, W, q, o = psi.compute_K((1,L), swapOp='F')
        Ks= 0
        print np.shape(W)
        for i in range(np.shape(W)[0]):
                kay = np.abs(W[i])*np.angle(W[i])*L/(2*np.pi)
                Ks += kay
                # int np.angle(W[i])
                
        K = np.angle(o[0])*L/(2*np.pi)

        print 'K =', K
                        
	data ={}

	data['chi'] = chi
	data['L'] = L
	data['DMRG_out'] = out
	data['psi'] = psi
	data['time'] = zeit
	data['model_par'] = model_par
	data['sim_par'] = sim_par

	return data

chi = args.chi
L = args.L

data = run()
