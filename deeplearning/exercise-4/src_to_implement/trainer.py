import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
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
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):

        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()  #clear old gradients from the last step
        # -propagate through the network
        self._model.train()
        predict_y = self._model(x)
        # -calculate the loss
        loss_val= self._crit(predict_y,y)  ## Loss function crit
        # -compute gradient by backward propagation
        loss_val.backward()   #derivative of the loss w.r.t. the parameters  bckp
        # -update weights
        self._optim.step()  #step based on the gradients of the parameters
        # -return the loss
        #return loss_val #error
        return float(loss_val)
        #TODO
    
    def val_test_step(self, x, y):
        # predict
        predict_y = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss_val = self._crit(predict_y, y)
        self._model.eval()
        # return the loss and the predictions
        return float(loss_val), predict_y
        #TODO
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self._model.train()
        self._optim.zero_grad()
        x=[]
        y=[]
        loss_for_epoch = 0.0
        for x ,y in tqdm(iter(self._train_dl)):  #batch and labels
            if self._cuda:
                x= x.cuda()
                y = y.cuda()
            x_tensor= t.tensor(x)
            y_tensor= t.tensor(y)
            loss = self.train_step(x_tensor ,y_tensor)
            loss_for_epoch = loss_for_epoch + loss
            print('----> Final Loss train: ', final_val)

        return loss_for_epoch / len(self._train_dl_)

    def calculate_tp(self):
        pass

    def calculate_fp(self):
        pass

    def calculate_tn(self):
        pass

    def calculate_fn(self):
        pass

    def calculate_precision(self, true_labels, predicted_labels):
        pass

    def calculate_recall(self, true_labels, predicted_labels):
        pass

    def calculate_accuracy(self, true_labels, predicted_labels):
        pass

    def calculate_f1score(self, true_labels, predicted_labels):
        pass
    
    def val_test(self):
        self._model.eval() #doubt
        self._optim.zero_grad()
        eval_loss = 0.0
        f1_total_metrics = 0
        for x, y in tqdm(iter(self._val_test_dl)):
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            x_tensor = t.tensor(x)
            y_tensor = t.tensor(y)
            loss, y_predicted = self.val_test_step(x_tensor,y_tensor)
            eval_loss = eval_loss + loss
            #f1 score calculation
        eval_loss = eval_loss / len(self._val_test_dl)
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these
        # metrics in designated functions
        # return the loss and print the calculated metrics
        return eval_loss
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        epoch_counter = 0
        validation_loss = []
        train_loss = []
        while True:
            break

            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
        return train_loss, validation_loss
                    
        
        
        