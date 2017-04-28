#!/usr/bin/env python
import sys
import math
import time
import numpy as np
import os
import matplotlib.pyplot as plt

#<====(MD code)====>#
#---Monika J. Williams---
#---July 2016---

#<====(General Functions)====>#

#Generates XYZ file to visualize crystal structure
def write_xyz(
    filename,
    atoms,
    symbol,
    append = False,
    newstep = True,
    natoms = 0.0,
    ):
    
    if append == False:
         fh = open(filename,"w")
    else:
        fh = open(filename,"a")

    if newstep == True:
        if natoms == 0.0:
             fh.write("%d\n\n" % (atoms.shape[0]))
        else:
             fh.write("%d\n\n" % (natoms))

        for A in range(atoms.shape[0]):
            fh.write("%s %14.6f %14.6f %14.6f\n" % (symbol,atoms[A,0],atoms[A,1],atoms[A,2]))
    else:
        for A in range(atoms.shape[0]):
            fh.write("%s %14.6f %14.6f %14.6f\n" % (symbol,atoms[A,0],atoms[A,1],atoms[A,2]))

#Generates fcc crystal structure
def generate_fcc(
    a = 1.0,
    L = 1,
    M = 1,
    N = 1,
    ):

    P = np.array([
        [0.0,0.0,0.0],
        [0.5,0.5,0.0],
        [0.5,0.0,0.5],
        [0.0,0.5,0.5],
        ])

    X = np.zeros((L*M*N*4,3))
    
    Tx = np.array([1,0,0])
    Ty = np.array([0,1,0])
    Tz = np.array([0,0,1])

    for l in range(L):
        for m in range(M):
            for n in range(N):
                lmn = l*M*N+m*N+n
                X[lmn*4:lmn*4+4,:] = P + l*Tx + m*Ty + n*Tz

    X *= a
    return X

#Calculates the energy and forces of a system
def compute_energy(
    atoms,
    a = 1.0,
    L = 1,
    M = 1,
    N = 1,
    eps = 1.0,
    sigma = 1.0,
    rc = 1.3,
    ):

    Lmax = int(math.ceil(rc/(a*L)))
    Mmax = int(math.ceil(rc/(a*M)))
    Nmax = int(math.ceil(rc/(a*N)))

    natoms = atoms.shape[0]
    rc2 = rc**2
    sigma2 = sigma**2
    eps2 = eps**2

    # Preallocation
    xi, xj = np.meshgrid(atoms[:,0],atoms[:,0],indexing='ij')
    yi, yj = np.meshgrid(atoms[:,1],atoms[:,1],indexing='ij')
    zi, zj = np.meshgrid(atoms[:,2],atoms[:,2],indexing='ij')
    dx = xi - xj
    dy = yi - yj
    dz = zi - zj

    E = 0.0
    F = np.zeros((natoms,3))
    for l in range(-Lmax,Lmax+1):
        for m in range(-Mmax,Mmax+1):
            for n in range(-Nmax,Nmax+1):
                Tx = L*a*l 
                Ty = M*a*m 
                Tz = N*a*n 

                r2 = (dx + Tx) * (dx + Tx) + (dy + Ty) * (dy + Ty) + (dz + Tz) * (dz + Tz) 
                v2 = sigma2 / r2
                v6 = v2 * v2 * v2
                v12 = v6 * v6
                E2 = v12 - v6
                E2[r2 > rc2] = 0.
                E2[r2 == 0.] = 0.
                E += np.sum(E2)
                G2 = (12*v12 - 6.*v6) / r2
                G2[r2 > rc2] = 0.
                G2[r2 == 0.] = 0.
                F[:,0] += np.sum(G2 * (dx + Tx), 1)
                F[:,1] += np.sum(G2 * (dy + Ty), 1)
                F[:,2] += np.sum(G2 * (dz + Tz), 1)

    E *= 2.0*eps
    F *= 4.0*eps
    return E,F

# ==> New Code <== #

def vv_step(
    x, # i
    v, # i 
    a, # i
    dt,
    a0,
    L,
    M,
    N,
    eps = 1.0,
    sigma = 1.0,
    rc = 1.3,
    ):

    xf = x + v * dt + 0.5 * a * dt**2
    # Put this all back in the box
    xf[:,0] = np.mod(xf[:,0] / (a0 * L),1) * a0 * L
    xf[:,1] = np.mod(xf[:,1] / (a0 * M),1) * a0 * M
    xf[:,2] = np.mod(xf[:,2] / (a0 * N),1) * a0 * N

    Vf, af = compute_energy(xf, a0, L, M, N, eps, sigma, rc)

    vf = v + 0.5 * (a + af) * dt

    Tf = 0.5 * np.sum(vf**2)
    Ef = Tf + Vf

    return Vf, Tf, Ef, xf, vf, af

