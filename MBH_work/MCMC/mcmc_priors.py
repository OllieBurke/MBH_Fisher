"""
God awful function to set up uniform priors across parameter space.
"""
import numpy as np
import sys

use_gpu = False
if use_gpu:
    import cupy as cp
    xp = cp
else:
    xp = np

M_low =  1e4 
M_high = 1e7

q_low = 0
q_high = 20

a_low = 0 
a_high = 1

inc_low  = 0.0
inc_high = np.pi

dist_low = 0.1
dist_high = 200

phi_ref_low = -2*np.pi 
phi_ref_high = 2*np.pi

lam_low = 0 
lam_high = 2*xp.pi

beta_low = -xp.pi/2
beta_high = xp.pi/2

psi_low = 0.0
psi_high = xp.pi

t_ref_low = 0
t_ref_high = 2*np.pi*1e7

def lprior_M(M,M_low, M_high):
    if M < M_low or M > M_high:
        print("M has failed")
        return -xp.inf
    else:
        return 0

def lprior_q(q,q_low, q_high):
    if q < q_low or q > q_high:
        print("q has failed")
        return -xp.inf
    else:
        return 0

def lprior_a1(a1,a_low, a_high):
    if a1 < a_low or a1 > a_high:
        print("a1 has failed")
        return -xp.inf
    else:
        return 0

def lprior_a2(a2,a_low, a_high):
    if a2 < a_low or a2 > a_high:
        print("a2 has failed")
        return -xp.inf
    else:
        return 0

def lprior_inc(inc,inc_low, inc_high):
    if inc < inc_low or inc > inc_high:
        print("inc has failed")
        return -xp.inf
    else:
        return 0
    
def lprior_dist(dist,dist_low, dist_high):
    if dist < dist_low or dist > dist_high:
        print("dist has failed")
        return -xp.inf
    else:
        return 0

def lprior_phi_ref(phi_ref,phi_ref_low, phi_ref_high):
    if phi_ref < phi_ref_low or phi_ref > phi_ref_high:
        print("phi_ref has failed")
        return -xp.inf
    else:
        return 0


def lprior_lam(lam, lam_low, lam_high):
    if lam < lam_low or lam > lam_high:
        print("the parameter", lam, "has failed")
        return -xp.inf
    else:
        return 0

def lprior_beta(beta, beta_low, beta_high):
    if beta < beta_low or beta > beta_high:
        print("the parameter", beta, "has failed")
        return -xp.inf
    else:
        return 0

def lprior_psi(psi, psi_low, psi_high):
    if psi < psi_low or psi > psi_high:
        print("the parameter", psi, "has failed")
        return -xp.inf
    else:
        return 0

def lprior_t_ref(t_ref, t_ref_low, t_ref_high):
    if t_ref < t_ref_low or t_ref > t_ref_high:
        print("the parameter", psi, "has failed")
        return -xp.inf
    else:
        return 0

def lprior(params):
    log_prior = (lprior_M(params[0],M_low, M_high)+ 
                lprior_q(params[1],q_low, q_high) +
                lprior_a1(params[2],a_low, a_high) + 
                lprior_a2(params[3],a_low, a_high) + 
                lprior_inc(params[4],inc_low, inc_high) +
                lprior_dist(params[5],dist_low,dist_high)  + 
                lprior_lam(params[6],lam_low, lam_high) +
                lprior_beta(params[7],beta_low, beta_high) +
                lprior_psi(params[8],psi_low, psi_high) +
                lprior_t_ref(params[9],t_ref_low, t_ref_high)) 
    if xp.isinf(log_prior):
        return -xp.inf
    return log_prior
