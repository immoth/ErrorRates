import numpy as np
from copy import deepcopy
import itertools
from fractions import Fraction as F

from models import dual_ising
from mps.mps import iMPS
from algorithms import simulation
from algorithms.linalg import np_conserved as npc
from tools.string import joinstr, to_mathematica_lists

def gcd(a, b):
	"""Greatest common divisor.  Return 0 if both a,b are zero, otherwise always return a non-negative number."""
	a = abs(a)
	b = abs(b)
	while b > 0:
		a, b = b, a % b      # after this, a > b
	return a

def lcm(a, b):
	if a == 0 and b == 0: return 0
	return a * b / gcd(a, b)


def model_GaudinYang(U, nu1, nu2, t1=1.0, t2=1.0):
	r"""
	H =  \sum_{a,s} t_s (c_a - c_{a+1})^\dag (c_a - c_{a+1})
		- \sum_{a,s} \mu_s c_a^\dag c_a
		+ U \sum_a n_{1,a} n_{2,a}
		+ (1/2) \sum_{a,s,s'} V_{s,s'} n_{s,a} n_{s',a}
	where we set \mu = 0, V = 0.
	
	Under the Jordan-Wigner transformation:
		c^\dag  <-->  S^- * string
		c^\dag c  <-->  1/2 - S^z.
	"""
	U, t1, t2 = float(U), float(t1), float(t2)
	hop_list = [ ('pXZ','pXI',-t1/2), ('pYZ','pYI',-t1/2), ('pIX','pZX',-t2/2), ('pIY','pZY',-t2/2) ]
	configL = lcm(nu1.denominator, nu2.denominator)
	if configL == 1: configL = 2
	config = [ np.ediff1d(np.rint(np.linspace(0, nu * configL, num = configL + 1, endpoint = True)).astype(int)) for nu in [nu1, nu2] ]
	Mpar = {
		'L': 1,
		'verbose': 1,
		'gzz': U,
		'hSz': -2 * t1 - U/2,
		'hTz': -2 * t2 - U/2,
		'constant offset': t1 + t2 + U/4,
		'extra_hoppings': [hop_list],
		'conserve_Sz': True,
		'conserve_diff_Sz': True,
		'dtype': float,
		'parstring': 't{},{}_U{}_nu{}o{},{}o{}'.format(t1, t2, U, nu1.numerator, nu1.denominator, nu2.numerator, nu2.denominator),
		'root config 1': config[0],
		'root config 2': config[1],
	}
	return Mpar

default_sim_par = {
	'VERBOSE': True,
	'STARTING_ENV_FROM_PSI': 1,
	'N_STEPS': 20,
	'MAX_ERROR_E': 1e-12,
	'MAX_ERROR_S': 1e-8,
	'MIN_STEPS': 40,
	'MAX_STEPS': 10000,
	'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'e_tol': 5*10**(-15), 'tol_to_trunc': 1/5.},
#	'mixer': (1e-3, 1.5, 5, 'id'),
}


def run_dmrg(sim, dmrg_par, model_par=None, chi=None, min_steps=None, save_sim=False):
	if sim is None:
		print 'Initializing "{}"...'.format(model_par['parstring'])
		M = dual_ising.dual_ising_model(model_par)

		##	Compute the initial state from the root configurations
		state_ordering = ['up', 'dn']		# 0 is spin up, 1 is spin dn
		initial_state = np.array([ M.states[0][state_ordering[s1]+state_ordering[s2]] for s1,s2 in itertools.izip(model_par['root config 1'], model_par['root config 2']) ])
		print joinstr(["Initial configs:  ", str(model_par['root config 1']) + '\n' + str(model_par['root config 2'])])
		psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve=M, bc='periodic')
		
		sim = simulation.simulation(psi, M)
		sim.model_par = model_par

	else:
		if model_par is not None:
			print 'Updating simulation "{}"...'.format(model_par['parstring'])
			sim.update_model(model_par)
		else:
			print 'Running simulation "{}"...'.format(model_par['parstring'])
		try:
			del sim.canon_psi
		except:
			pass
	
	sim_par = deepcopy(dmrg_par)
	if chi is not None: sim_par['CHI_LIST'] = {0:chi}
	if min_steps is not None: sim_par.update['MIN_STEPS'] = min_steps
	sim.dmrg_par = sim_par
	print 'DMRG parameters:\n' + '\n'.join([ "  {} : {}".format(k,v) for k,v in sim.dmrg_par.items() ])
	sim.ground_state()

	try:
 		sim.append
	except AttributeError:
		sim.append = {}
	if 'xi' in sim.append: del sim.append['xi']

	sim.append['GS Energy'] = sim.sim_stats[-1]['Es'][-1]
	sim.canon_psi = sim.psi.copy()
	sim.canon_psi.canonical_form()
	if save_sim:
		filename = outroot + model_par['parstring'] + '_chi{}'.format(max(sim_par['CHI_LIST'].values()))
		uncanon_psi = sim.psi
		sim.psi = sim.canon_psi
		print 'Presaving simulation to "{}"...'.format(filename)
		sim.save(filename)
	sim.append['xi'] = sim.canon_psi.correlation_length()
	print "xi = {}".format(sim.append['xi'])
	if save_sim:
		print 'Saving simulation to "{}"...'.format(filename)
		sim.save(filename)
		sim.psi = uncanon_psi
	print
	return sim


def load_sim(model_par, chi, verbose=1):
	filename = outroot + model_par['parstring'] + '_chi' + str(chi)
	print 'Loading "{}"...'.format(filename)
	try:
		sim = simulation.simulation.load(filename, dual_ising.dual_ising_model, force_mod_verbose=0)
	except IOError, e:
		if verbose >= 1: print "  IOError!", e
		return None
	return sim


def measure_correlator(sim):
	if hasattr(sim, 'canon_psi'):
		psi = sim.canon_psi
	else:
		psi = sim.psi
	M = sim.M
	xi = sim.append['xi']
	dist = int(5 * xi)

	site_n1 = 0.5 - psi.site_expectation_value(M.Sz)
	site_n2 = 0.5 - psi.site_expectation_value(M.Tz)
	print "Occupation number  <n1> = {},  <n2> = {}".format(site_n1, site_n2)
	corr_c1 = psi.correlation_function(M.SmZ, M.SpI, dist, OpStr=M.pZZ)
	corr_c2 = psi.correlation_function(M.ISm, M.ZSp, dist, OpStr=M.pZZ)
#	corr_c1c2 = psi.correlation_function(M.SmSm, M.SpSp, dist)
	print to_mathematica_lists(corr_c1)
	print to_mathematica_lists(corr_c2)
#	print to_mathematica_lists(corr_c1c2)
	print


##################################################

np.set_printoptions(linewidth=2000, precision=5, threshold=4000)
outroot = 'GaudinYang/'		# this determines where everything is saved/loaded.

model_par = model_GaudinYang(-2.0, F(1,7), F(1,7), t1=1.0, t2=1.0)

if 1:		# run simulation and save
	sim_par = deepcopy(default_sim_par)
	CHI_LIST = dict([(0,14), (20,20), (40,28), (80,40), (140,57), (240,80), (400,113)][:3])
	sim_par.update({'CHI_LIST':CHI_LIST, 'MIN_STEPS':1.3*max(CHI_LIST.keys())})
	sim = run_dmrg(None, sim_par, model_par=model_par, save_sim=False)
	measure_correlator(sim)
if 0:
	sim = load_sim(model_par, 40)		# load simulation from disk
	measure_correlator(sim)
if 0:
	data_table = []
	for chi in [20,28,40,57,80,113]:
		sim = load_sim(model_par, chi)
		if sim is None: continue
		data_table.append([chi, np.mean(sim.psi.entanglement_entropy()), sim.append['xi']])
		print sim.append['GS Energy']
		print np.abs(np.fft.fft( 0.5-sim.psi.site_expectation_value(sim.M.Sz) ))
		print np.abs(np.fft.fft( 0.5-sim.psi.site_expectation_value(sim.M.Tz) ))
	print to_mathematica_lists(data_table)
