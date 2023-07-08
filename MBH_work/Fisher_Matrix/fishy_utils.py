
import numpy as np
from lisatools.utils.constants import PC_SI 
from bbhx.waveformbuild import BBHWaveformFD

use_gpu = False
if use_gpu:
    import cupy as cp
    xp = cp
else:
    xp = np

wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu = use_gpu)

def MBH_f(M,q,a1,a2,inc,dist_Gpc,phi_ref,lam,beta,psi,t_ref, **kwargs):
    """
    Code to generate massive black holes. Change of parametrisation to M, q and distance
    in Gpc. Outputs TDI A, E T for MBH. 
    """
    freq = kwargs['freq']
    f_ref = kwargs['f_ref'] 
    modes = kwargs['modes']

    m1 = q*M/(1.0 + q)
    m2 = M/(1.0 + q)
    dist_m = dist_Gpc * 1e9 * PC_SI
    MBH_AET = wave_gen(m1,m2,a1,a2,dist_m,phi_ref,f_ref,inc,lam,beta,psi,t_ref, freqs=freq, modes=modes, direct=False, fill=True, squeeze=True, length=len(freq))[0]
    return MBH_AET

def build_fish_matrix(M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref, **kwargs):
    """
    Build the fisher matrix. Inputs parameters, outputs fisher matrix (NOT covariance matrix)
    """
    delta_f = kwargs["delta_f"]
    PSD_AET = kwargs["PSD"] 

    params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref])
    N_params = len(params)

    steps = np.array([1, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]) # Steps for numerical derivative. Seem "fine"
    N_params = len(steps) 
    deriv_vec = []
    params_copy = params.copy()
    for j in range(N_params):
        params[j] = params[j] + steps[j] # this is the f(x + h) step
        h_f_p = MBH_f(*params, **kwargs) 

        params[j] = params[j] - 2 * steps[j] # this is the f(x - h) step
        h_f_m = MBH_f(*params, **kwargs)

        deriv_h_f = (h_f_p - h_f_m) / (2*steps[j]) # compute derivative
        deriv_vec.append(deriv_h_f)
        params = params_copy # reset parameters

    # For build matrices to place values
    gamma_A, gamma_E = xp.eye(N_params), xp.eye(N_params)

    for i in range(N_params):
        for j in range(i,N_params):
            gamma_A[i,j] = 4*delta_f * xp.real(xp.sum((deriv_vec[i][0]*xp.conjugate(deriv_vec[j][0]) / (PSD_AET[0]))))
            gamma_E[i,j] = 4*delta_f * xp.real(xp.sum((deriv_vec[i][1]*xp.conjugate(deriv_vec[j][1]) / (PSD_AET[1]))))

    # Build diagonal matrices 
    gamma_A_diag = xp.diag(xp.diag(gamma_A)) 
    gamma_E_diag = xp.diag(xp.diag(gamma_E))

    # Subtract off first element of diagonal matrix
    gamma_A = gamma_A - 0.5*gamma_A_diag
    gamma_E = gamma_E - 0.5*gamma_E_diag

    # Build symmetric matrix. Notice that diagonal is twice that of gamma_(AE) [restoring consistency]
    gamma_A = gamma_A + gamma_A.T
    gamma_E = gamma_E + gamma_E.T

    # Compute joint FM over A and E. 
    gamma_AE = gamma_A + gamma_E 
    return gamma_AE
