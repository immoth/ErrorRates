import numpy as np
from copy import deepcopy
import itertools
from fractions import Fraction as F

from models import dual_ising
from mps.mps import iMPS
from algorithms import simulation
from algorithms.linalg import np_conserved as npc
from tools.string import joinstr, to_mathematica_lists

from algorithms import DMRG
from algorithms import TEBD

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

U=-2.0
nu1=F(1,7)
nu2=F(1,7)
t1=1.0
t2=1.0

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

M = dual_ising.dual_ising_model(Mpar)

state_ordering = ['up', 'dn']
initial_state = np.array([ M.states[0][state_ordering[s1]+state_ordering[s2]] for s1,s2 in itertools.izip(Mpar['root config 1'], Mpar['root config 2']) ])

psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve=M, bc='finite')


"""
"DMRG Stuff"
#args = sys.argv[1:]
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

LP = DMRG.ground_state(psi,M,sim_par)

print(LP[1])
"""

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


print(M.U[0])


