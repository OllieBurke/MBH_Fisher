from poplar.nn.networks import load_model
from poplar.nn.training import train, train_test_split
from poplar.nn.rescaling import ZScoreRescaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

device = "cuda:0"

parameters = pd.read_csv("../Fisher_Matrix/data/1e6.csv")
fish_data = np.load("../Fisher_Matrix/data/1e6_fish.npy")

try:
    keeplim = np.where(fish_data[:,0] ==0)[0][0]
except:
    keeplim = fish_data.shape[0]
print(f"Chucking after the {keeplim}'th sample")

fish_data = fish_data[:,:-1] / fish_data[:,-1][:,None] **2

xdata = torch.as_tensor(parameters.to_numpy()[:keeplim], device=device)
ydata = torch.as_tensor(fish_data[:keeplim][:,:-1])

# we symlog the data
signs = torch.sign(ydata)
ydata = signs * torch.log(torch.abs(ydata))

train_fraction = 0.99

xtrain, xtest, ytrain, ytest = train_test_split([xdata, ydata], train_fraction)

# define the neural network
model = load_model("models/snr_res_f1e-5_symlog/model.pth")
model.set_device(device)

ypred = model.run_on_dataset(xtest)

percerrs = np.log10(abs(ypred/ytest).cpu().numpy())
# list_params = ['M','q','a1','a2','inc', 'dist_Gpc', 'phi_ref', 'lambda', 'beta', 'psi', 't_ref']

plt.figure(dpi=200)
for i in range(fish_data.shape[1]-1):
    plt.hist(percerrs[:,i], bins='auto', density=True, histtype="step")#, label=list_params[i])
# plt.legend()
plt.xlabel(r'$\log_{10}(\mathrm{Percent Error})$')
# plt.yscale('log')
plt.savefig(f'models/{model.name}/performance_hist.png')
plt.close()

ypred_unlog = torch.sign(ypred) * torch.exp(torch.abs(ypred))
ytest_unlog = torch.sign(ytest) * torch.exp(torch.abs(ytest))

percerrs = abs(1 - ytest_unlog/ypred_unlog).cpu().numpy()
plt.figure(dpi=200)
for i in range(fish_data.shape[1]-1):
    plt.plot(np.sort(percerrs[:,i]), np.linspace(0, 1, percerrs.shape[0], endpoint=False))
plt.xlabel('Fractional error |1 - truth/pred|')
plt.ylabel('CDF')
plt.xscale('log')
plt.savefig(f'models/{model.name}/empirical_cdf_unlog.png')