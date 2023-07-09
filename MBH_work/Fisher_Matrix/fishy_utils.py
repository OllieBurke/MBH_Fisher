use_gpu = True
if use_gpu:
    import cupy as cp
    xp = cp
    xp.cuda.runtime.setDevice(0)

else:
    xp = np

import numpy as np
from lisatools.utils.constants import PC_SI 
from bbhx.waveformbuild import BBHWaveformFD


wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu = use_gpu)

def inner_prod(signal_1_f,signal_2_f,PSD, delta_f):
    return 4*delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj()/PSD))

def MBH_f(M,q,a1,a2,inc,dist_Gpc,phi_ref,lam,beta,psi,t_ref, **kwargs):
    """
    Code to generate massive black holes. Change of parametrisation to M, q and distance
    in Gpc. Outputs TDI A, E T for MBH. 
    """
    # breakpoint()
    freq = kwargs['freq']
    f_ref = kwargs['f_ref'] 
    modes = kwargs['modes']

    m1 = q*M/(1.0 + q)
    m2 = M/(1.0 + q)
    dist_m = dist_Gpc * 1e9 * PC_SI
    MBH_AET = xp.squeeze(wave_gen(m1,m2,a1,a2,dist_m,phi_ref,f_ref,inc,lam,beta,psi,t_ref, freqs=freq, modes=modes, direct=False, fill=True, squeeze=True, length=len(freq)))
    return MBH_AET

def build_fish_matrix(M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref, return_sparse=False, **kwargs):
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

    if return_sparse:  # sparse return, save some time not building matrices and in correct form for neural net
        gamma_sparse = xp.zeros(N_params * (N_params + 1) // 2)
        cnt = 0 # i'm lazy
        for i in range(N_params):  # TODO: rewrite these inner products to vectorise over [A, E]
            for j in range(i,N_params):
                gamma_sparse[cnt] += 4*delta_f * xp.real(xp.sum((deriv_vec[i][0]*xp.conjugate(deriv_vec[j][0]) / (PSD_AET[0]))))
                gamma_sparse[cnt] += 4*delta_f * xp.real(xp.sum((deriv_vec[i][1]*xp.conjugate(deriv_vec[j][1]) / (PSD_AET[1]))))
                cnt += 1
        return gamma_sparse
        
    else:    # For build matrices to place values

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


def vectorised_fisher_matrix(M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref, **kwargs):
    """
    Build the fisher matrix. Inputs parameters, outputs fisher matrix (NOT covariance matrix)
    """
    import time

    delta_f = kwargs["delta_f"]
    PSD_AET_arr = kwargs["PSD_ARR"] 

    params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref])
    steps = np.array([1, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]) # Steps for numerical derivative. Seem "fine"
    N_params = len(params)  

    # there are 2*Nparam waveforms to evaluate. Get them all at once
    Nderivs = N_params * 2
    # params_copy = params.copy()
    step_eye = np.eye(N_params)*steps
    all_params = np.ones((Nderivs, N_params))
    all_params[:N_params] = (params + step_eye)
    all_params[N_params:] = (params - step_eye)
    # breakpoint()
    all_waves = MBH_f(*all_params.T, **kwargs)[:,:2,:]  # (Nderivs, 2, len) we discard T here to save time later
    deriv_vec = (all_waves[:N_params] - all_waves[N_params:]) / (2*xp.asarray(steps))[:,None,None] # compute 
    Nfishelems = N_params * (N_params + 1) // 2

    gamma_sparse = xp.zeros(Nfishelems)

    cnt = 0 # i'm lazy
    for i in range(N_params):  # I expect we can do this without looping, too, by constructing indexing arrays carefully.
        for j in range(i,N_params):
            gamma_sparse[cnt] =  \
                xp.sum(
                    xp.real(
                        xp.sum(
                            deriv_vec[i]*xp.conjugate(deriv_vec[j]) / PSD_AET_arr[None,:,:], axis=-1  # first sum over L
                        )
                    ), 
                axis=-1)  # second sum over A,E
            cnt += 1
    
    gamma_sparse *= (4*delta_f)
    return gamma_sparse