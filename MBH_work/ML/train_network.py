from poplar.nn.networks import LinearModel
from poplar.nn.training import train, train_test_split
from poplar.nn.rescaling import ZScoreRescaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

device = "cuda:1"

parameters = pd.read_csv("../Fisher_Matrix/vectorised_params_fixlim_big.csv")
fish_data = np.load("../Fisher_Matrix/vectorised_fisher_values_fixlim_big.npy")

try:
    keeplim = np.where(fish_data[:,0] ==0)[0][0]
except:
    keeplim = fish_data.shape[0]
print(f"Chucking after the {keeplim}'th sample")
xdata = torch.as_tensor(parameters.to_numpy()[:keeplim], device=device)
ydata = torch.as_tensor(fish_data[:keeplim][:,:-1])

# define a rescaler, which handles the scaling of input data to facilitate training
rescaler = ZScoreRescaler(xdata, ydata)#, yfunctions=[torch.log, torch.exp])

train_fraction = 0.99

xtrain, xtest, ytrain, ytest = train_test_split([xdata, ydata], train_fraction)

# define the neural network
model = LinearModel(
    in_features=xdata.shape[1],
    out_features=fish_data.shape[1]-1,
    neurons=[128, ] * 6,
    activation=torch.nn.SiLU,
    rescaler=rescaler,
    name="1e6"
).double()

model.set_device(device)

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Training points: ", xtrain.shape[0])

train(
    model,
    data=[xtrain, ytrain, xtest, ytest],
    n_epochs=10000,
    n_batches=20,
    loss_function=torch.nn.L1Loss(),
    optimiser=optimiser,
    update_every=1000,
    verbose=True,
)

ypred = model.run_on_dataset(xtest)
percerrs = np.log10(np.abs((1 - ypred/ytest).cpu().numpy()))
# list_params = ['M','q','a1','a2','inc', 'dist_Gpc', 'phi_ref', 'lambda', 'beta', 'psi', 't_ref']

plt.figure(dpi=200)
for i in range(fish_data.shape[1]-1):
    plt.hist(percerrs[:,i], bins='auto', density=True, histtype="step")#, label=list_params[i])
# plt.legend()
plt.xlabel(r'$\log_{10}(\mathrm{Percent Error})$')
plt.savefig(f'models/{model.name}/performance_hist.png')
plt.xlim(-3, 3)
plt.savefig(f'models/{model.name}/performance_hist_zoom.png')
