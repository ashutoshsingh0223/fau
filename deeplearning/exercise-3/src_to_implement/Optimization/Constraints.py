

class L1_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        pass

    def norm(self, weights):
        pass


class L2_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        pass

    def norm(self, weights):
        pass
