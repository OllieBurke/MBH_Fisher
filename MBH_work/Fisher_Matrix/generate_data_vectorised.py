use_gpu = True
if use_gpu:
    import cupy as cp
    xp = cp
    xp.cuda.runtime.setDevice(0)
else:
    xp = np

from fishy_utils import MBH_f, build_fish_matrix, inner_prod, vectorised_fisher_matrix

import os
import numpy as np
import matplotlib.pyplot as plt
from bbhx.waveformbuild import BBHWaveformFD

from lisatools.sensitivity import get_sensitivity
from lisatools.utils.constants import PC_SI, YRSID_SI

from tqdm import tqdm # progress bar
import pandas as pd
from multiprocessing import Pool

N_channels = 2
channel = ["A","E"]
sens_fn_calls = ["noisepsd_AE","noisepsd_AE"]


wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu = use_gpu)

nevents = int(1e5)

# set parameters

# we're just going to use uniform distributions on most of these, except log-uniform on mass and distance

M = 10**np.random.uniform(4., 8., nevents) # total mass: IMPORTANT, this is the redshifted mass, source frame implies cosmology which we might want to infer.
q = np.random.uniform(1., 10., nevents) # mass ratio (m1/m2)
a1 = np.random.uniform(0., 1., nevents) # Spin parameter of body 1
a2 = np.random.uniform(0., 1., nevents) # Spin parameter of body 2
inc = np.random.uniform(0, np.pi, nevents) # inclination

dist_Gpc = 10**np.random.uniform(0, np.log10(50), nevents)  # Distance in Gpc
phi_ref = np.random.uniform(0, 2*np.pi, nevents) # phase at f_ref
lam = np.random.uniform(0, 2*np.pi, nevents)  # ecliptic longitude
beta = np.random.uniform(-np.pi/2, np.pi/2, nevents)  # ecliptic latitude
psi = np.random.uniform(0,np.pi, nevents)  # polarization angle
t_ref = np.random.uniform(0, 4, nevents) * YRSID_SI  # t_ref  (in the SSB reference frame)  not sure about this

f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f)) - This is just internal, not a parameter really?
modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref]).T 
list_params = ['M','q','a1','a2','inc', 'dist_Gpc', 'phi_ref', 'lambda', 'beta', 'psi', 't_ref']

N_params = len(params[0])
fish_arr = np.zeros((nevents, 1 + N_params * (N_params + 1) // 2))  # last column is the SNR!
event_data = pd.DataFrame(params, columns=list_params)

# test save
event_data.to_csv("./vectorised_params.csv",index=False)
np.save("./vectorised_fisher_values.npy", fish_arr)

delta_f = 1e-5                  
freq = xp.arange(1e-4,1e-1,delta_f)

kwargs = {"freq" : freq,
          "delta_f" : delta_f,
          "f_ref" : f_ref,
          "modes" : modes}
PSD_AET = [get_sensitivity(freq, sens_fn = PSD_choice) for PSD_choice in sens_fn_calls]
kwargs['PSD'] = PSD_AET
kwargs['PSD_ARR'] = xp.array(PSD_AET)

# def get_snr_and_fish(params_here):   
#     MBH_AET = MBH_f(*params_here, **kwargs)
#     outputs = np.zeros(1 + N_params * (N_params + 1) // 2)
#     SNR2_AET = xp.asarray([inner_prod(MBH_AET[i],MBH_AET[i],PSD_AET[i],delta_f) for i in range(N_channels)])
#     total_snr = xp.sum(SNR2_AET)**(1/2)
#     outputs[-1] = total_snr

#     gamma_AE = build_fish_matrix(*params_here, return_sparse=True, **kwargs)
#     outputs[:-1] = gamma_AE#.get()
#     return outputs

# with Pool(20) as p:
#     results = list(tqdm(p.imap(get_snr_and_fish, params), total=nevents))

# calculate the SNRs first

batch_size = 500
nbatches = nevents // batch_size
import time
for batchnum in tqdm(range(nbatches)):
    params_here = params[batch_size*batchnum:batch_size*(batchnum+1)]    
    MBH_AET = MBH_f(*params_here.T, **kwargs)[:,:2,:]  # (batch_size, 2, len)
    total_snrs = xp.sum(
            4*delta_f*xp.real(
                xp.sum(MBH_AET * MBH_AET.conj()/kwargs["PSD_ARR"][None,:,:], axis=-1  # sum over len
            )
        ), axis=-1  # sum over A, E
    )**0.5
    fish_arr[batch_size*batchnum:batch_size*(batchnum+1), -1] = total_snrs.get()


for event_num in tqdm(range(nevents)):

    # gamma_AE = build_fish_matrix(*params_here, return_sparse=True, **kwargs)
    gamma_AE_fast = vectorised_fisher_matrix(*params_here, **kwargs)
    # print(gamma_AE)
    # print(gamma_AE_fast)
    # print(gamma_AE_fast / gamma_AE)
    # breakpoint()
    fish_arr[event_num, :-1] = gamma_AE_fast.get()

    if event_num % 1000 == 0:
        np.save("./vectorised_fisher_values.npy", fish_arr)

np.save("./vectorised_fisher_values.npy", fish_arr)
