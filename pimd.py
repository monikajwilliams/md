#!/usr/bin/env python
import sys
import math
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import md


#<====(MD code)====>#
#---Monika J. Williams---
#---July 2016---


def rp_coords(
    atoms,      #classical cartesian coordinates for particles in system
    P = 5,      #number of beads
    ):

    worlds = []
    for p in range(P):
        worlds.append(atoms)
    
    return worlds

def evolve_ring(
    q,               #positions
    p,               #momenta
    m,               #masses
    dt = 0.01,        #timestep
    w_P = 0.001,   #force constant
    a0 = math.sqrt(2.0)*2.0**(1.0/6.0),
    L = 2,
    M = 2,
    N = 2,
    ):

    q = np.array(q)
    p = np.array(p)
    P = len(q) #number of beads
    natoms = len(q[1]) #number of classical atoms

    new_q = np.zeros(np.shape(q)) 
    new_p = np.zeros(np.shape(p)) 

    for n in range(natoms):
        ring_q = q[:,n,:]
        ring_p = p[:,n,:]

        normqx = np.fft.fftn(ring_q[:,0]) #converting to normal modes
        normpx = np.fft.fftn(ring_p[:,0]) #converting to normal modes
        
        normqy = np.fft.fftn(ring_q[:,1]) #converting to normal modes
        normpy = np.fft.fftn(ring_p[:,1]) #converting to normal modes

        normqz = np.fft.fftn(ring_q[:,2]) #converting to normal modes
        normpz = np.fft.fftn(ring_p[:,2]) #converting to normal modes

        new_modq = np.zeros((P,3),dtype=complex)        
        new_modp = np.zeros((P,3),dtype=complex)        

        for l in range(P):

            LL = float(l+1)
            PP = float(P)
            W_l = 2.0*w_P*np.sin((LL*math.pi)/PP)
            Hm = np.zeros((2,2))
            Hm[0,0] += math.cos(W_l*dt)
            Hm[0,1] += -m[n]*(W_l*math.sin(W_l*dt))
            Hm[1,0] += (1.0/(m[n]*W_l))*math.sin(W_l*dt)
            Hm[1,1] += math.cos(W_l*dt)
            
            vecx = np.array([normpx[l],normqx[l]])
            vecy = np.array([normpy[l],normqy[l]])
            vecz = np.array([normpz[l],normqz[l]])
            
            #new_modp[l,0],new_modq[l,0] = np.dot(vecx,Hm)
            #new_modp[l,1],new_modq[l,1] = np.dot(vecy,Hm)
            #new_modp[l,2],new_modq[l,2] = np.dot(vecz,Hm)

            new_modp[l,0],new_modq[l,0] = np.dot(Hm,vecx)
            new_modp[l,1],new_modq[l,1] = np.dot(Hm,vecy)
            new_modp[l,2],new_modq[l,2] = np.dot(Hm,vecz)

        new_q[:,n,0] = np.fft.ifftn(new_modq[:,0]) 
        new_p[:,n,0] = np.fft.ifftn(new_modp[:,0]) 
        new_q[:,n,1] = np.fft.ifftn(new_modq[:,1]) 
        new_p[:,n,1] = np.fft.ifftn(new_modp[:,1]) 
        new_q[:,n,2] = np.fft.ifftn(new_modq[:,2]) 
        new_p[:,n,2] = np.fft.ifftn(new_modp[:,2]) 

        # Put this all back in the box
        new_q[:,n,0] = np.mod(new_q[:,n,0] / (a0 * L),1) * a0 * L
        new_q[:,n,1] = np.mod(new_q[:,n,1] / (a0 * M),1) * a0 * M
        new_q[:,n,2] = np.mod(new_q[:,n,2] / (a0 * N),1) * a0 * N


    return new_q,new_p
