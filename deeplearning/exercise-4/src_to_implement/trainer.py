import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
# from evaluation import create_evaluation
import numpy as np
from tqdm import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimiser
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_cb=None):  # The stopping criterion.
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl

        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb

        if self._cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        # TODO
        self._model.train()
        y_pred = self._model(x)
        self._optim.zero_grad()
        loss = self._crit(y_pred, y)
        loss.backward()
        self._optim.step()

        return float(loss)

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        # TODO
        self._model.eval()
        y_pred = self._model(x)
        loss = self._crit(y_pred, y)

        return float(loss), y_pred

    def train_epoch(self):

    # set training mode
    # iterate through the training set
    # transfer the batch to "cuda()" -> the gpu if a gpu is given
    # perform a training step
    # calculate the average loss for the epoch and return it
    # TODO
        loss = 0
        for x, y in tqdm(iter(self._train_dl)):
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            loss += self.train_step(x, y)

        loss = loss / len(self._train_dl)

        return loss

    def val_test(self):

    # set eval mode
    # disable gradient computation
    # iterate through the validation set
    # transfer the batch to the gpu if given
    # perform a validation step
    # save the predictions and the labels for each batch
    # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
    # return the loss and print the calculated metrics
    # TODO
        #self._model.mode = 'val'
        loss = list()

        for x, y in tqdm(iter(self._val_test_dl)):
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            with t.no_grad():
                loss_var, _ = self.val_test_step(x, y)
                loss.append(loss_var)

        avg_loss = np.sum(loss) / len(loss)
        avg_loss = float( avg_loss)

        print('----> Average Loss: ', avg_loss)

        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO
        train_loss = list()
        val_loss = list()
        counter_epochs = 0

        while True:
            counter_epochs += 1

    # stop by epoch number
    # train for a epoch and then calculate the loss and metrics on the validation set
    # append the losses to the respective lists
    # use the save_checkpoint function to save the model for each epoch
    # check whether early stopping should be performed using the early stopping callback and stop if so
    # return the loss lists for both training and validation
    # TODO
            if counter_epochs > epochs:
               break

            print('--> Epoch:', counter_epochs)

            train_loss.append( self.train_epoch() )
            avg_loss = self.val_test()
            val_loss.append( avg_loss )

            if self._early_stopping_cb is not None:
                self._early_stopping_cb.step(avg_loss)
                if self._early_stopping_cb.should_stop():
                    best_epoch = counter_epochs - self._early_stopping_cb.patience - 1
                    print('The best epoch is ', best_epoch)
                    self.save_checkpoint(best_epoch)
                    break

        return  train_loss, val_loss

