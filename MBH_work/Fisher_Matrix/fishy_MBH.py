import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from bbhx.waveformbuild import BBHWaveformFD
from fishy_utils import MBH_f, build_fish_matrix

from lisatools.sensitivity import get_sensitivity
from lisatools.utils.constants import PC_SI, YRSID_SI

N_channels = 2
channel = ["A","E"]
sens_fn_calls = ["noisepsd_AE","noisepsd_AE"]

use_gpu = True
if use_gpu:
    xp = cp
else:
    xp = np

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

def inner_prod(signal_1_f,signal_2_f,PSD, delta_f):
    return 4*delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj()/PSD))

SNR2_AET = xp.asarray([inner_prod(MBH_AET[i],MBH_AET[i],PSD_AET[i],delta_f) for i in range(N_channels)])

for i in range(N_channels):
    print("For channel {}, we observe SNR = {}".format(channel,SNR2_AET[i]**(1/2)))

print("Total SNR for A, E, T is given by", xp.sum(SNR2_AET)**(1/2))
# ==================== Fisher matrix =======================#

gamma_AE = build_fish_matrix(*params, **kwargs)

param_cov_AE = np.linalg.inv(gamma_AE)

delta_theta = np.sqrt(np.diag(param_cov_AE))
list_params = ['M','q','a1','a2','inc', 'dist_Gpc', 'phi_ref', 'lambda', 'beta', 'psi', 't_ref']

for i in range(N_params):
    print("Relative precision in parameter {0} = {1}".format(list_params[i],delta_theta[i]))