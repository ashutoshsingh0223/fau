import torch as t
from torch import nn
from src_to_implement.data import ChallengeDataset
from src_to_implement.trainer import Trainer
from src_to_implement.EarlyStopping import EarlyStopping

from matplotlib import pyplot as plt
import numpy as np
from src_to_implement.model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split

epochs = 50
stop_limit = 5
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
data = pd.read_csv('data.csv').to_numpy()
train_data, validation_data = train_test_split(data, shuffle=False, test_size=0.2, random_state=42)
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_dataset = t.utils.data.DataLoader(train_data, batch_size=12 )
print('Loading train dataset ')
validation_dataset = t.utils.data.DataLoader(validation_data, batch_size=12)
print('Loading validation dataset')
# create an instance of our ResNet model
# TODO
model = ResNet()
print('Loading Resnet model')
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
criterion_loss = nn.BCELoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.0001)

patience = 5
early_stopping = EarlyStopping(patience=patience)

trainer = Trainer(model=model, crit=criterion_loss, optim=optimizer,
                  train_dl=train_dataset, val_test_dl=validation_dataset,
                  cuda=True, early_stopping_patience=early_stopping.patience, early_stopping_crit=early_stopping)

# go, go, go... call fit on trainer
res = trainer.fit(epochs=epochs)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')