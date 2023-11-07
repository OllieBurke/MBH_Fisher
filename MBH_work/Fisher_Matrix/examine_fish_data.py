import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


parameters = pd.read_csv("vectorised_params_fixlim_big.csv")
fish_data = np.load("vectorised_fisher_values_fixlim_big.npy")

try:
    keeplim = np.where(fish_data[:,0] ==0)[0][0]
except:
    keeplim = fish_data.shape[0]

parameters = parameters.iloc[:keeplim]
fish_data = fish_data[:keeplim]

# normalise by inverse square of the SNR
fish_data_plot = fish_data[:,:-1] / fish_data[:,-1][:,None] **2

for i in range(fish_data_plot.shape[1]):
    print(i)
    # plt.hist(np.log10(abs(fish_data_plot[:,i])), 'auto')
    plt.hist(fish_data_plot[:,i], 10000)
    plt.yscale('log')
    plt.savefig(f"fishhists/{i}.png")
    plt.close()
