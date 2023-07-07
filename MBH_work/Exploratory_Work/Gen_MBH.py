import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from bbhx.waveformbuild import BBHWaveformFD

from lisatools.sensitivity import get_sensitivity
from lisatools.utils.constants import PC_SI, YRSID_SI

use_gpu = True
if use_gpu:
    xp = cp
else:
    xp = np
wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu = use_gpu)

# set parameters
f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
phi_ref = 0.0 # phase at f_ref
m1 = 1e6
m2 = 1e6
a1 = 0.2
a2 = 0.4
dist = 18e3  * PC_SI * 1e6 # 3e3 in Mpc
inc = np.pi/3.
beta = np.pi/4.  # ecliptic latitude
lam = np.pi/5.  # ecliptic longitude
psi = np.pi/6.  # polarization angle
t_ref = 1.0 * YRSID_SI  # t_ref  (in the SSB reference frame)

# Spacing in frequency. Equals 1/T_obs, linear spacing
delta_f = 1e-5                  
freq = cp.arange(1e-4,1e-1,delta_f)

# Declare number of modes to use 
modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

MBH_AET = wave_gen(m1, m2, a1, a2,
                          dist, phi_ref, f_ref, inc, lam,
                          beta, psi, t_ref, freqs=freq,
                          modes=modes, direct=False, fill=True, squeeze=True, length=len(freq))[0]


N_channels = 3
channel = ["A","E","T"]
sens_fn_calls = ["noisepsd_AE","noisepsd_AE","noisepsd_T"]

PSD_AET = [get_sensitivity(freq, sens_fn = PSD_choice) for PSD_choice in sens_fn_calls]

def inner_prod(signal_1_f,signal_2_f,PSD, delta_f):
    return 4*delta_f * xp.real(sum(signal_1_f * signal_2_f.conj()/PSD))

SNR2_AET = xp.asarray([inner_prod(MBH_AET[i],MBH_AET[i],PSD_AET[i],delta_f) for i in range(N_channels)])

for i in range(N_channels):
    print("For channel {}, we observe SNR = {}".format(channel,SNR2_AET[i]**(1/2)))

print("Total SNR for A, E, T is given by", xp.sum(SNR2_AET)**(1/2))

# =================== Signal processing ============================
T_obs = 1/delta_f
# Frequencies are missing zeroth frequency (in theory). We can resolve 
# up to -(-N/2), so need N time points. 
N_t = 2*(len(freq) - 1) 
delta_t = ((N_t) * delta_f)**-1
print("Sampling interval is",delta_t)

wave_A_td = (delta_t)**-1 * xp.fft.irfft(MBH_AET[0]) # Be careful, numpys convention!

# ====================== Plot the signal ===========================
t = np.arange(0,len(wave_A_td)*delta_t, delta_t)
wave_A_td_np = xp.asnumpy(wave_A_td)

os.chdir('/home/ad/burkeol/work/MBH_work/Exploratory_Work/plots')
plt.plot(t/60/60,wave_A_td_np, label = 'TD waveform')
plt.xlabel(r'Time [hours]',fontsize = 16)
plt.ylabel(r'h^{(A)}(t)',fontsize = 16)
plt.title(r'Response function of MBH - A channel', fontsize = 16)
plt.tight_layout()
plt.savefig("MBH_TD.pdf")
plt.clf()
