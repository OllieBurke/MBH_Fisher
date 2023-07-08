import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from bbhx.waveformbuild import BBHWaveformFD
sys.path.append("../Fisher_Matrix")
from fishy_utils import MBH_f, build_fish_matrix

from lisatools.sensitivity import get_sensitivity
from lisatools.utils.constants import PC_SI, YRSID_SI
import h5py
import emcee


N_channels = 2
channel = ["A","E"]
sens_fn_calls = ["noisepsd_AE","noisepsd_AE"]

use_gpu = False
if use_gpu:
    import cupy as cp
    xp = cp
else:
    xp = np

def inner_prod(signal_1_f,signal_2_f,PSD, delta_f):
    return 4*delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj()/PSD))

##======================Likelihood and Posterior (change this)=====================

def llike(params):
    waveform_prop_f = MBH_f(*params, **kwargs)

    diff_f_AET = [data_f_AET[k] - waveform_prop_f[k] for k in range(N_channels)]
    inn_prod = xp.asarray([inner_prod(diff_f_AET[k],diff_f_AET[k],PSD_AET[k],delta_f) for k in range(N_channels)])
    return(-0.5 * xp.sum(inn_prod))

def lpost(params):
    '''
    Compute log posterior
    '''
    if xp.isinf(lprior(params)):
        print("Prior returns -\infty")
        return -np.inf
    else:
        return llike(params)

wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu = use_gpu)

# set parameters
M = 2e6 # total mass
q = 3.0 # mass ratio (m1/m2)
a1 = 0.2 # Spin parameter of body 1
a2 = 0.4 # Spin parameter of body 2
inc = np.pi/3. # inclination

dist_Gpc = 20.0  # Distance in Gpc
phi_ref = np.pi # phase at f_ref
lam = np.pi/5.  # ecliptic longitude
beta = np.pi/4.  # ecliptic latitude
psi = np.pi/6.  # polarization angle
t_ref = 1.0 * YRSID_SI  # t_ref  (in the SSB reference frame)

f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f)) - This is just internal, not a parameter really?

# Set number of modes
modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref]) 
N_params = len(params)

# Frequency spacing, delta_f = 1/T_obs
delta_f = 1e-5                  
freq = xp.arange(1e-4,1e-1,delta_f)

kwargs = {"freq" : freq,
          "delta_f" : delta_f,
          "f_ref" : f_ref,
          "modes" : modes}

MBH_AET = MBH_f(*params, **kwargs)

PSD_AET = [get_sensitivity(freq, sens_fn = PSD_choice) for PSD_choice in sens_fn_calls]

kwargs['PSD'] = PSD_AET

SNR2_AET = xp.asarray([inner_prod(MBH_AET[i],MBH_AET[i],PSD_AET[i],delta_f) for i in range(N_channels)])

for i in range(N_channels):
    print("For channel {}, we observe SNR = {}".format(channel,SNR2_AET[i]**(1/2)))

print("Total SNR for A, E, T is given by", xp.sum(SNR2_AET)**(1/2))

# ==================== INPUT NOISE ======================================
# variance_noise_AET = [N_t * PSD_AET[k] / (4*delta_t) for k in range(N_channels)]
# noise_f_AET_real = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
# noise_f_AET_imag = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
# noise_f_AET = xp.asarray([noise_f_AET_real[k] + 1j * noise_f_AET_imag[k] for k in range(N_channels)])
# =======================================================================

data_f_AET = MBH_AET 

np.random.seed(1234)
##===========================MCMC Settings (change this)============================
iterations = 3000 #10000  # The number of steps to run of each walker
burnin = 0 # I burn results when I process the data in a jupyter notebook. 
nwalkers = 50  #50 #members of the ensemble, like number of chains

d = 1 

# Set starting parameters. Starting near true parameters, not doing a search. 
start_M = M*(1. + d * 1e-4 * np.random.randn(nwalkers,1))   
start_q = q*(1. + d * 1e-4 * np.random.randn(nwalkers,1))
start_a1 = a1*(1. + d * 1e-4 * np.random.randn(nwalkers,1))
start_a2 = a2*(1. + d * 1e-4 * np.random.randn(nwalkers, 1))
start_inc = inc*(1. + d * 1e-4 * np.random.randn(nwalkers, 1))
start_dist_Gpc = dist_Gpc*(1. + d * 1e-4 * np.random.randn(nwalkers, 1))

start_phi_ref = phi_ref*(1. + d * 1e-4 * np.random.randn(nwalkers,1))
start_lam = lam*(1. + d * 1e-4 * np.random.randn(nwalkers,1))
start_beta = beta*(1. + d * 1e-4 * np.random.randn(nwalkers,1))
start_psi = psi*(1. + d * 1e-4 * np.random.randn(nwalkers,1))

start_t_ref = t_ref*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))

start = np.hstack((start_M,start_q, start_a1, start_a2, start_inc, start_dist_Gpc, start_phi_ref, start_lam, start_beta, start_psi, start_t_ref))

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]
from mcmc_priors import lprior   # Input prior knowledge
print("Should be zero if there is no noise", lpost(start[0])) # Useful check.

os.chdir('mcmc_results')
moves_stretch = emcee.moves.StretchMove(a=2)  
fp = "test_MBH_mcmc.h5" 
backend = emcee.backends.HDFBackend(fp)
#start = backend.get_last_sample() #Continue
backend.reset(nwalkers, ndim) #Start New

if use_gpu:
    sampler = emcee.ensemblesampler(nwalkers, ndim, lpost, 
                                    backend=backend, moves = moves_stretch)
    sampler.run_mcmc(start,iterations, progress = True, tune=True)
else:
    # If not on a GPU, we can use multiprocessing. 
    from multiprocessing import (get_context,cpu_count) 
    N_cpus = cpu_count()
    pool = get_context("fork").Pool(N_cpus)        # M1 chip -- allows multiprocessing
    sampler = emcee.ensemblesampler(nwalkers, ndim, lpost, pool = pool,  
                                    backend=backend, moves = moves_stretch)

    sampler.run_mcmc(start,iterations, progress = True, tune=True)
