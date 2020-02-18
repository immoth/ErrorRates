import time
import string
import sys
import os
import copy
import numpy as np
import cPickle
import cProfile
import json
from mps.mps import iMPS
from algorithms import DMRG
from algorithms import DMRG_core
from algorithms.linalg import np_conserved as npc
from algorithms.linalg import npc_helper as npc_helper
from models import quantum_hall as mod
from models import qh_model_wf
from tools.string import to_mathematica_lists
from tools.string import uni_to_str

import matplotlib.pyplot as plt
import matplotlib.cm as cm
	
args = sys.argv[1:]
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

v = Bunch(**{"chi_min": 128, "p": 1, "q": 3, "tL": [0.0], "Vs": {"haldane": [0.0, 0.447, 0., 0.], "Dmax":None, "eps":0.01, "TK": [0.0]}, "chi_inc": 0.2,  "svd_max": 18, "trunc_cut":10**(-8.), "LxL": [22.0], "chi_max": 256})

		
cons_cm = False

def run():

	print "Potential parameters:", v.Vs

	(p, q) = v.p, v.q
	
	model_par  = {
				'particle_type': 'F',
            	'filling_factor' : (p, q),
				'cons_cm':cons_cm,
				'fracture_mpo': False,
				't':[0.],
				'Vs': v.Vs #See note on defs above			  
			}
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


#Initial states and 'grouping' info. We group 'grp' sites together, with 'l*rep' total ungrouped sites in unit cell. If you want to fiddle with grouping behavior this is the only code that requires modification.


	if (p, q) == (1, 5):
		initial_state = np.array( [0, 0, 1, 0, 0])
		grp = 5
		rep = 2
	if (p, q) == (2, 5):
		initial_state = np.array( [0, 1, 0, 1, 0])
		grp = 5	
		rep = 2
	if (p, q) == (1, 3):                                          
		initial_state = np.array( [0, 1, 0]*4)
		grp = 4
		rep = 1
	if (p, q) == (1, 2):
		initial_state = np.array( [0, 1, 1, 0])
		grp = 4
		rep = 2
	if (p, q) == (2, 4):
		initial_state = np.array( [0, 1, 1, 0]*3)
		grp = 3
		rep = 1

	np.set_printoptions(precision=3, suppress = True)

	def iterate(psi, Lx, t, Vm, chi):
	
		paramstr = "Lx = {}, chi = {}".format(Lx,  chi)
		print '-'*30 + 'begin run ' + paramstr + '-'*30
		
		model_par['Vs']['haldane'] = Vm
		model_par['Lx'] = Lx
		M = mod.QH_model(model_par)
		m = M.Vmr
		print
		print "Vmr at Lx = ", Lx
		print m/np.abs(np.max(m)) #Useful to see how it is truncated
		#print M.V_mask
		M.increase_L(l)
		M.combine_sites(grp)
		#M.fracture_mpo = False
		if firstRun:
			bounds = [ np.min([chi, 2**i]) for i in range(4, 9)] + [chi]
			sim_par['CHI_LIST'] = dict( (i, bounds[i]) for i in range(6))
		else:
			sim_par['CHI_LIST'] = {0:chi}
			
		start = time.time()
		truncation, Estat, Sstat, RP, LP = DMRG.ground_state(psi,M,sim_par)
		print "DMRG took ", time.time() - start, "Seconds.\n"	
		E0 = Estat[-1]
		print "E0:", E0
		psi.canonical_form2(form = 'B', verbose=1)
		print "Xi", psi.correlation_length()
		
		
		#[0, 1, 0, 0] [1, 0, 0, 1] [0, 0, 1, 0]
		print "---"*10 + "Trying some defects!" + "---"*10
		psiQP = DMRG.defect_ground_state(psi, LP, RP, psi, LP, RP, [0, 3], [1, 4], M, sim_par, charge = np.array([1,17]))
		
		psiQP = psiQP.split_sites()
		psi = psi.split_sites()
		Ns = psiQP.site_expectation_value(M.N)
		
		print Ns
		print psi.site_expectation_value(M.N)
		
		#xs = np.arange(0, Lx, 0.25)
		xs = [0.]
		ys = np.arange(2., (psiQP.L - 1.)*M.kappa-2., 0.1)
		rho = np.abs(M.measure_density(psiQP, xs, ys))*2*np.pi-1./3.
		print rho
		plt.plot(ys,rho)
		plt.show()
		plt.plot(ys+0.05, np.cumsum(rho)*0.1/M.kappa)
		#im = plt.imshow(rho, interpolation='bilinear', cmap=cm.gray,
		#origin='lower', extent = [ys[0], ys[-1], xs[0], xs[-1]])
		plt.show()
		
		quit()
		"""
		M.plot_spectrum(psi, 'absmin', bond = 1)
		w, q_ind = DMRG_core.zero_site_half_spectrum(RP[1], M.vL[1])
		ws = [w]
		q_inds = [q_ind]
		"""

		
		e0s = []
		for r in range(len(LP)):
			e0s.append( DMRG_core.zero_site_bulk_energy(LP[r], RP[r], psi.s[r]) )
		
		print "e0s", e0s
		
		ws = [None]*len(RP)
		q_inds = [None]*len(RP)
		for r in range(len(RP)):
			ws[r], q_inds[r] = DMRG_core.zero_site_bulk_spectrum(LP[0], RP[r].imap_Q(M.translate_Q, -r))

		
		egs = e0s[0] - LP[0].age*E0*grp - RP[0].age*E0*grp
		
		for r in range(len(RP)):
			ws[r] = ws[r] - LP[0].age*E0*grp - RP[r].age*E0*grp - egs

		def find_mu(w, q_ind):
			min_q = 10.**10.
			min_nq = 10.**10.
			min_0 = 10.**10.
			for qi in q_ind:
				if qi[2]==q:
					min_q = np.min( [np.min(w[qi[0]:qi[1]]), min_q])
				if qi[2]==-q:
					min_nq = np.min( [np.min(w[qi[0]:qi[1]]), min_nq])
				if qi[2]==0:
					min_0 = np.min( [np.min(w[qi[0]:qi[1]]), min_0])
			w-=min_0
			return 0.5*(min_q - min_nq)/q
			
		def print_e(w, q_ind, mu):
			print
			print
			n = q_ind[0, 2]

			k = []
			e = []
			for q in q_ind:
				if q[2]!=n:
					
					k = np.array(k)
					e = np.array(e)
					s = np.argsort(e)
					k = k[s]
					e = e[s] - mu*n
					print "N, k, e", n, k[0], e[0]
					#print "K", k
					#print "E", e
					plt.plot(k, e, 'o')
					k = []
					e = []
					n = q[2]
					
				e.extend( list(w[q[0]:q[1]]))
				k.extend([q[-1]]*(q[1] - q[0]))
			
			k = np.array(k)
			e = np.array(e)
			s = np.argsort(e)
			k = k[s]
			e = e[s] - mu*n
	
			print "N, k, e", n, k[0], e[0]
			plt.plot(k, e, 'o')
			plt.show()
		
		mu = find_mu(ws[0], q_inds[0])
		print "mu:", mu
		for r in range(len(ws)):
			print_e(ws[r], q_inds[r], mu)

		sim_par['RP'] = RP
		sim_par['LP'] = LP
	

	l = len(initial_state)
	model_par['Lx']=np.max(v.LxL)	
	M = mod.QH_model(model_par)
	M.increase_L(l)
	psi = iMPS.product_imps(M.d, initial_state, dtype=np.float, conserve = M, form = 'B')
	psi.combine_sites(grp)
	psi.increase_L(psi.L*rep)
	
	M.combine_sites(grp)
	print "MPO dim", M.H_mpo[0].shape
	print "MPO stats", M.H_mpo[0].sparse_stats()
	M.check_sanity()


	T0 = time.time()
	firstRun = True

	iterate(psi, 14., 0., [0., 0.447, 0., 0.1*0.277], 64)

	

	print "Total Time", time.time() - T0
	
if __name__ == "__main__":

	#npc_helper.test_create()
	#quit()
	run()
	#cProfile.run('run()', 'PyNumberHunt')
	
