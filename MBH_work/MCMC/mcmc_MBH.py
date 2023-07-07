import os
import sys
import numpy as np
import cupy as cp
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

use_gpu = True
if use_gpu:
    xp = cp
else:
    xp = np

def inner_prod(signal_1_f,signal_2_f,PSD, delta_f):
    return 4*delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj()/PSD))

##======================Likelihood and Posterior (change this)=====================

def llike(params):
    # M_val = float(params[0])
    # q_val =  float(params[1])
    # a1_val =  float(params[2])            # This works fine! 
    # a2_val = float(params[3])
    # inc_val = float(params[4])
    # dist_Gpc_val = float(params[5])
    # phi_ref_val = float(params[6])
    # lam_val = float(params[7])
    # beta_val = float(params[8])
    # psi_val = float(params[9])
    # t_ref_val = float(params[10])

    waveform_prop_f = MBH_f(*params, **kwargs)

    diff_f_AET = [data_f_AET[k] - waveform_prop_f[k] for k in range(N_channels)]
    inn_prod = xp.asarray([inner_prod(diff_f_AET[k],diff_f_AET[k],PSD_AET[k],delta_f) for k in range(N_channels)])
    return(-0.5 * xp.sum(inn_prod))

def lpost(params):
    '''
    Compute log posterior
    '''
    if cp.isinf(lprior(params)):
        print("Prior returns -\infty")
        return -np.inf
    else:
        return llike(params)

wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu = use_gpu)

# set parameters
M = 2e6
q = 3.0
a1 = 0.2
a2 = 0.4
inc = np.pi/3.

dist_Gpc = 20.0 #18e3  * PC_SI * 1e6 # 3e3 in Mpc
phi_ref = 0.0 # phase at f_ref
lam = np.pi/5.  # ecliptic longitude
beta = np.pi/4.  # ecliptic latitude
psi = np.pi/6.  # polarization angle
t_ref = 1.0 * YRSID_SI  # t_ref  (in the SSB reference frame)

f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f)) - This is just internal, not a parameter really?
modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref]) 
N_params = len(params)

delta_f = 1e-5                  
freq = cp.arange(1e-4,1e-1,delta_f)

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

# variance_noise_AET = [N_t * PSD_AET[k] / (4*delta_t) for k in range(N_channels)]
# noise_f_AET_real = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
# noise_f_AET_imag = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
# noise_f_AET = xp.asarray([noise_f_AET_real[k] + 1j * noise_f_AET_imag[k] for k in range(N_channels)])

data_f_AET = MBH_AET 

##===========================MCMC Settings (change this)============================
iterations = 40000 #10000  # The number of steps to run of each walker
burnin = 0
nwalkers = 50  #50 #members of the ensemble, like number of chains

# n = 0
d = 0 

#here we should be shifting by the *relative* error! 

start_M = M*(1. + d * 1e-3 * np.random.randn(nwalkers,1))   # changed to 1e-6 careful of starting points! Before I started on secondaries... haha.
start_q = q*(1. + d * 1e-3 * np.random.randn(nwalkers,1))
start_a1 = a1*(1. + d * 1e-3 * np.random.randn(nwalkers,1))
start_a2 = a2*(1. + d * 1e-3 * np.random.randn(nwalkers, 1))
start_inc = inc*(1. + d * 1e-3 * np.random.randn(nwalkers, 1))
start_dist_Gpc = dist_Gpc*(1. + d * 1e-3 * np.random.randn(nwalkers, 1))

start_phi_ref = phi_ref*(1. + d * 1e-3 * np.random.randn(nwalkers,1))
start_lam = lam*(1. + d * 1e-3 * np.random.randn(nwalkers,1))
start_beta = beta*(1. + d * 1e-3 * np.random.randn(nwalkers,1))
start_psi = psi*(1. + d * 1e-3 * np.random.randn(nwalkers,1))

start_t_ref = t_ref*(1. + d * 1e-3 * np.random.randn(nwalkers, 1))

start = np.hstack((start_M,start_q, start_a1, start_a2, start_inc, start_dist_Gpc, start_phi_ref, start_lam, start_beta, start_psi, start_t_ref))

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]

print("Should be zero if there is no noise", llike(start[0]))
breakpoint()
from mcmc_func_forwards import *   # Input prior knowledge

os.chdir('/home/ad/burkeol/work/MBH_work/MCMC/mcmc_results')
moves_stretch = emcee.moves.StretchMove(a=2)  
fp = "test_MBH_mcmc.h5" 
backend = emcee.backends.HDFBackend(fp)
start = backend.get_last_sample() #Continue
#backend.reset(nwalkers, ndim) #Start New

sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, 
                                backend=backend, moves = moves_stretch)

sampler.run_mcmc(start,iterations, progress = True, tune=True)

