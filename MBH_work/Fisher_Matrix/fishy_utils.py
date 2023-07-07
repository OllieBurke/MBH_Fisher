
import cupy as cp
from matplotlib.cbook import file_requires_unicode
import numpy as np
from lisatools.sensitivity import get_sensitivity
from lisatools.utils.constants import PC_SI, YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD

use_gpu = True
if use_gpu:
    xp = cp
else:
    xp = np

wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu = use_gpu)

def MBH_f(M,q,a1,a2,inc,dist_Gpc,phi_ref,lam,beta,psi,t_ref, **kwargs):
    freq = kwargs['freq']
    f_ref = kwargs['f_ref'] 
    modes = kwargs['modes']

    m1 = q*M/(1.0 + q)
    m2 = M/(1.0 + q)
    dist_m = dist_Gpc * 1e9 * PC_SI
    MBH_AET = wave_gen(m1,m2,a1,a2,dist_m,phi_ref,f_ref,inc,lam,beta,psi,t_ref, freqs=freq, modes=modes, direct=False, fill=True, squeeze=True, length=len(freq))[0]
    return MBH_AET

def build_fish_matrix(M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref, **kwargs):
    delta_f = kwargs["delta_f"]
    PSD_AET = kwargs["PSD"] 

    params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref])
    N_params = len(params)
    MBH_AET = MBH_f(*params, **kwargs)

    steps = np.array([1, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    N_params = len(steps) 
    deriv_vec = []
    params_copy = params.copy()
    for j in range(N_params):
        params[j] = params[j] + steps[j]
        h_f_p = MBH_f(*params, **kwargs) 

        params[j] = params[j] - 2 * steps[j]
        h_f_m = MBH_f(*params, **kwargs)

        deriv_h_f = (h_f_p - h_f_m) / (2*steps[j])
        deriv_vec.append(deriv_h_f)
        params = params_copy

    gamma_A, gamma_E = xp.eye(N_params), xp.eye(N_params)

    for i in range(N_params):
        for j in range(i,N_params):
            if i == j:
                gamma_A[i,j] = 2*delta_f * xp.real(xp.sum((deriv_vec[i][0]*xp.conjugate(deriv_vec[j][0]) / (PSD_AET[0]))))
                gamma_E[i,j] = 2*delta_f * xp.real(xp.sum((deriv_vec[i][1]*xp.conjugate(deriv_vec[j][1]) / (PSD_AET[1]))))
            else:
                gamma_A[i,j] = 4*delta_f * xp.real(xp.sum((deriv_vec[i][0]*xp.conjugate(deriv_vec[j][0]) / (PSD_AET[0]))))
                gamma_E[i,j] = 4*delta_f * xp.real(xp.sum((deriv_vec[i][1]*xp.conjugate(deriv_vec[j][1]) / (PSD_AET[1]))))

    gamma_A = gamma_A + gamma_A.T
    gamma_E = gamma_E + gamma_E.T

    gamma_AE = gamma_A + gamma_E 
    return gamma_AE
