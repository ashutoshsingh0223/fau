class EarlyStopping():
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self,
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvement before terminating.
            the counter be reset after each improvement
        """
        # self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e15
        self.stopped_epoch = 0


    def end_training(self, epoch, current_loss):
        if current_loss is None:
            return False
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
                return False
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    return True
                self.wait += 1


class EarlyStoppingCallback:

    def __init__(self, patience):
        #initialize all members you need
        self.patience = patience
        self.counter = 0
        self.prevBest = 1000

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress
        if current_loss < self.prevBest:
            self.prevBest = current_loss
            self.counter = 0
        else:
            self.counter += 1

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        if self.counter > self.patience:
            return True
        else:
            return False