from poplar.nn.networks import LinearModel
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

# define a rescaler, which handles the scaling of input data to facilitate training
rescaler = ZScoreRescaler(xdata, ydata)#, yfunctions=[torch.log, torch.exp])

train_fraction = 0.99

xtrain, xtest, ytrain, ytest = train_test_split([xdata, ydata], train_fraction)

# define the neural network
model = LinearModel(
    in_features=xdata.shape[1],
    out_features=fish_data.shape[1]-1,
    neurons=[256, ] * 20,
    activation=torch.nn.SiLU,
    rescaler=rescaler,
    name="snr_res_f1e-5_symlog"
).double()

model.set_device(device)

optimiser = torch.optim.Adam(model.parameters(), lr=5e-6)

print("Training points: ", xtrain.shape[0])

train(
    model,
    data=[xtrain, ytrain, xtest, ytest],
    n_epochs=10000,
    n_batches=500,
    loss_function=torch.nn.L1Loss(),
    optimiser=optimiser,
    update_every=1000,
    verbose=True,
)

ypred = model.run_on_dataset(xtest)
print(ypred)
percerrs = np.log10(abs(ypred/ytest).cpu().numpy())
# list_params = ['M','q','a1','a2','inc', 'dist_Gpc', 'phi_ref', 'lambda', 'beta', 'psi', 't_ref']
print(percerrs)
breakpoint()
plt.figure(dpi=200)
for i in range(fish_data.shape[1]-1):
    plt.hist(percerrs[:,i], bins='auto', density=True, histtype="step")#, label=list_params[i])
# plt.legend()
plt.xlabel(r'$\log_{10}(\mathrm{Percent Error})$')
# plt.yscale('log')
plt.savefig(f'models/{model.name}/performance_hist.png')
plt.close()

percerrs = abs(ypred/ytest - 1).cpu().numpy()
plt.figure(dpi=200)
for i in range(fish_data.shape[1]-1):
    plt.plot(np.sort(percerrs[:,i]), np.linspace(0, 1, percerrs.shape[0], endpoint=False))
plt.xlabel('Fractional error (pred/truth)')
plt.ylabel('CDF')
plt.xscale('log')
plt.savefig(f'models/{model.name}/empirical_cdf.png')