### Machine Learning code explanation

Training/testing data is split with `train_test_split`: I chose 99% here as the split ratio but this can be adjusted. We appear to be data-limited here so probably want to keep this as high as possible.

The model is constructed here:
```
model = LinearModel(
    in_features=xdata.shape[1],
    out_features=fish_data.shape[1]-1,
    neurons=[128, ] * 6,
    activation=torch.nn.SiLU,
    rescaler=rescaler,
    name="1e6"
).double()
```

You can adjust a few things here, namely the layer structure (here it is 6 layers of 128, but it can be anything - note that the number of parameters increases quadratically with the layer size!), activation function (I use a SiLU to have a continuous output, but other options are available too - see the `pytorch` documentation) and the name (which is just used for file handling purposes)

The learning rate (which adjusts how quickly the model shifts its parameters during training) can be changed in the `lr` kwarg to the `optim` object, it's currently set to `1e-4` which should be reasonable.

Lastly, we can adjust some features of training
```
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
```
Here we can change `n_epochs` to adjust the number of cycles through the training set we make, `n_batches` to adjust how many batches we feed the dataset in as each time (`n_batches=1` means the entire dataset so 1 batch per epoch). Loss function describes what we minimise the network against for training, `L1Loss` and `MSELoss` are both good choices. Lastly `update_every` just controls how often the loss plot is updated and the model is saved. `save_best` can be used to specifically keep the model that performed best on the test data, which is good for stopping early before overfitting occurs.

The model is evaluated on some input data with `model.run_on_dataset(...)`.