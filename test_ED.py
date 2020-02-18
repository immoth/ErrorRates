#Taken from majorana_chain_example_idmrg.py
import numpy as np
import scipy.special
import scipy.linalg as lng
import copy
from models import spin_chain as mod 
from mps.mps import iMPS
from algorithms import DMRG
from algorithms import TEBD
from algorithms.linalg import np_conserved as npc
from algorithms.linalg import npc_helper
from tools.string import joinstr
from algorithms import simulation

import time
import cProfile
import sys
import os

s0=np.array([[1,0],[0,1]])
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])

def Sx(l,L):
    if(l==0):
        stp=sx
        for i in range(1,L):
            stp=np.kron(s0,stp)
    else:
        stp=s0
        for i in range(1,l):
            stp=np.kron(s0,stp)
        stp=np.kron(sx,stp)
        for i in range(l+1,L):
            stp=np.kron(s0,stp)
    return stp

def Sy(l,L):
    if(l==0):
        stp=sy
        for i in range(1,L):
            stp=np.kron(s0,stp)
    else:
        stp=s0
        for i in range(1,l):
            stp=np.kron(s0,stp)
        stp=np.kron(sy,stp)
        for i in range(l+1,L):
            stp=np.kron(s0,stp)
    return stp

def Sz(l,L):
    if(l==0):
        stp=sz
        for i in range(1,L):
            stp=np.kron(s0,stp)
    else:
        stp=s0
        for i in range(1,l):
            stp=np.kron(s0,stp)
        stp=np.kron(sz,stp)
        for i in range(l+1,L):
            stp=np.kron(s0,stp)
    return stp



def Mdot(list):
    mtp=list[0]
    for i in range(1,len(list)):
        mtp=np.dot(mtp,list[i])
    return mtp



def H(hz,Jx,Jy,Jz,L):
    htp=0
    for l in range(0,L):
        htp=htp+hz*Sz(l,L)
    for l in range(0,L-1):
        htp=htp+Jx*np.dot(Sx(l,L),Sx(l+1,L))+Jy*np.dot(Sy(l,L),Sy(l+1,L))+Jz*np.dot(Sz(l,L),Sz(l+1,L))
    return htp

L=4
hz=-0.05/2
Jx=-1.0/4
Jy=0.0/4
Jz=0.0/4
H0=H(hz,Jx,Jy,Jz,L)

[E0,psi_0]=np.linalg.eig(H0)
idx = E0.argsort()
psi_0 = np.transpose(psi_0[:,idx])
E0 = E0[idx]


print(E0)

t=5.0
U=lng.expm(1j*H0*t)
Ub=lng.expm(-1j*H0*t)
psi_t=np.dot(U,psi_0[0])



print(np.dot(np.conjugate(psi_t),psi_0[0]))
print(abs(np.dot(np.conjugate(psi_t),psi_0[0]))**2)


print(np.dot(np.conjugate(psi_t),np.dot(H0,psi_t)))



