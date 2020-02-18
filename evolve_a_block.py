from algorithms.linalg import np_conserved as npc
import numpy as np
from numpy import random
from mps import mpo
from models import boson2d as mod
from mps.mps import iMPS
import pylab as pl
import time
from matplotlib import cm




Lx = 5
Ly = 5
chi_max = 120
dt = 0.05
N_steps = 20
var_par = { 'N_update_env':1, 'max_iter':1, 'min_iter':1}
# try increasing max_iter to make sure VAR converged. We need to write code to auto detect this at some point.

V = 0. #nearest neighbor V
t = 1. #hopping.

bc = 'open'

model_par = { 'd':2, 
    'nu':(0, 1), 
    'Lx':Lx,
	'Ly':Ly,
    't': [ [ (1, 0, t), (0, 1, t) ] ]*int(Lx - 1 + (bc=='periodic')) + [[(0, 1, t)]]*int(bc=='open'),
    #'U':[ [ (1, 0, V), (0, 1, V) ] ]*int(Lx - 1 + (bc=='periodic')) + [[(0, 1, V)]]*int(bc=='open'),
	#'mu': 3*(np.random.random(Lx*Ly)-0.5), #disorder
    'verbose': 0}

truncation_par = {'chi_max': chi_max,
                  'trunc_cut'  : 1e-9,
                  'svd_max'  : 18.}

np.set_printoptions(linewidth=2000, precision=5, threshold=4000)
M = mod.boson2d_model(model_par)


B = 1
initial_state = [ (B <= i/Ly < Ly-B) and  (B <= i%Ly < Ly-B) for  i in range(Lx*Ly)]

H = mpo.MPO.mpo_from_W(M.H_mpo, M.vL, M.vR, bc=M.bc)


#Us = [H.make_U(-1j*dt) ] #first order
Us = [ H.make_U(-1j*dt*(1+1j)/2.), H.make_U(-1j*dt*(1-1j)/2.)] #2nd order

print("i am printing: "+str(M.H_mpo[3]))


"""
psi = iMPS.product_imps(M.d, initial_state, dtype=complex, conserve = M, bc='finite')


N = psi.site_expectation_value(M.N).real
N = N.reshape((-1, Lx))
t0 = time.time()

Ns = []

for i in range(N_steps):
	N = psi.site_expectation_value(M.N).real
	N = N.reshape((-1, Lx))
	Ns.append(N)
	if i < Lx:
		for U in Us:
			U.apply_mpo(psi,compression='SVD', truncation_par=truncation_par)
		print ".",
	else:
		for U in Us:
			U.apply_mpo(psi,compression='VAR', truncation_par=truncation_par, var_par=var_par)
		print ".",

print "Final anisotropy under mirror:", np.linalg.norm(Ns[-1] - Ns[-1].T)
cnt = 0
for j in range(0,len(Ns),len(Ns)/4+1):
	fig = pl.subplot(1, 4, cnt+1)

	for i in range(Lx):
		pl.plot([i,i],[-0.5, Ly-0.5],'k')
		pl.plot([-0.5, Lx-0.5],[i,i],'k')

	for x in range(Lx):
		for y in range(Ly):
			n = 1-Ns[j][x,y]
			
			pl.plot(x,y,'o',color = [n,n,n], markersize=12)
			t = j*dt
			pl.title('$t = %.2f$'%t)
 
	pl.xlim([-1,Lx])
	pl.ylim([-1,Ly])
	pl.gca().set_aspect(1)
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)

	cnt += 1

pl.show()
"""
