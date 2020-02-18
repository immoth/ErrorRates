import numpy as np
import time, sys, os

from models import long_range_spin_chain as mod
from algorithms import simulation
from mps.mps import iMPS
from mps import mpo
from algorithms.linalg import np_conserved as npc
from matplotlib import pyplot as plt

np.set_printoptions(linewidth=np.inf, precision=8, threshold=np.inf, suppress=True)

L = 11

chi_max = 32
dt = 0.025 #size of time step
N_steps = 120 #

N_iter = 2 #number of variational compression steps
compression='VAR'

#Truncation parameters during time evolution
truncation_par = {'chi_max': chi_max,
	'trunc_cut': 1e-8,
		'svd_max'  : 18.}


xi = L/2. #capture the power law out to xi
numJ = 3 * xi

#We cut off the power law in the form J
N = np.array(np.arange(1, numJ + 1), float)
J =  1./N**2/(1+np.exp((N-xi)/10.))
Jcoupling = np.hstack([[0], J]) #onsite terms is set to 0

model_par = {
	'verbose': 2,
	'L': 1,
	'S': 0.5,
	#'Jz': [0.],
	'Jpm': Jcoupling,
	'Veps': 1e-6,
	'ignore_herm_conj':False
}

M = mod.spin_chain_model(model_par)
H = mpo.MPO.mpo_from_W(M.H_mpo, M.vL, M.vR,bc='finite')

print '\tMPO.chi =', M.chi
psi = iMPS.product_imps(M.d, [M.up, M.dn]*(L/2) + [M.up]*np.mod(L, 2), dtype=np.float, conserve = M, bc='finite')

sim = simulation.simulation(psi, M)
sim.model_par = model_par
sim.dmrg_par = {
		#'CHI_LIST': {0:10, 10:20, 30:28, 60:40, 100:57, 200:80, 300:95},
		'CHI_LIST': {0:chi_max},
		'N_STEPS': 10,
		'STARTING_ENV_FROM_PSI': 1,
		'MIN_STEPS': 30,
		'MAX_STEPS': 100,
		'MAX_ERROR_E' : 1e-8,
		'MAX_ERROR_S' : 1e-5,
		'LANCZOS_PAR': {'N_min': 2, 'N_max': 20, 'p_tol': 1e-6, 'p_tol_to_trunc': 1/100., 'cache_v': np.inf},
		
	}

#First find the ground state
sim.ground_state()

E0 = sim.sim_stats[-1]['Es'][-1]/L
psi.canonical_form()
psi0 = sim.psi.copy()
Sz = psi.site_expectation_value(M.Sz)
plt.plot(Sz)

#Flip a spin; apply S+ operator
B = psi.getB(L/2)
B = npc.tensordot(M.Sp[0], B, axes = ['p*', 'p'])
psi.setB(L/2, B)
psi.canonical_form()

Sz = psi.site_expectation_value(M.Sz)
plt.plot(Sz)
plt.xlabel('x')
plt.ylabel(r'$S^z$')
plt.title(r'$S^z$ before and after spin flip')
plt.show()

#Make the 2nd order MPOs
H = mpo.MPO.mpo_from_W(M.H_mpo, M.vL, M.vR,bc='finite')
U1 = H.make_U(1j*dt*(1.+1j)/2., E0 = E0)
U2 = H.make_U(1j*dt*(1.-1j)/2., E0 = E0)
psi.astype(np.complex)

S  = []
Szs = []
O = []

#Alternate the two time steppers
for i in range(N_steps):
	print ".",
	Sz = psi.site_expectation_value(M.Sz).real
	Szs.append(Sz)
	U1.apply_mpo(psi,truncation_par=truncation_par, compression=compression, max_iterations=N_iter)
	U2.apply_mpo(psi,truncation_par=truncation_par, compression=compression, max_iterations=N_iter)
	O.append(psi.overlap(psi0))
	S.append(psi.entanglement_entropy()[L/2])

print
plt.plot(np.arange(N_steps)*dt, S)
plt.title('Entanglement @ L /2')
plt.show()

plt.plot(np.arange(N_steps)*dt, np.abs(O))
plt.plot(np.arange(N_steps)*dt, 0.5+np.angle(O)/2./np.pi)
plt.title('Overlap')
plt.ylim([-0.1, 1.1])
plt.show()

Szs = np.array(Szs)
plt.imshow(Szs, interpolation='nearest', extent = [0, L, dt*N_steps, 0])
plt.gca().set_aspect('auto')
plt.colorbar()
plt.title(r'$\langle S^m(0, L/2) S^z(t, x) S^p(0, L/2) \rangle_0$')
plt.ylabel(r'$t$')
plt.xlabel(r'$x$')
plt.show()
