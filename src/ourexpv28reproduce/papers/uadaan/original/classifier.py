import torch
import torch.nn as nn
import bayesian as bnn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=16, out_features=4),
            nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=4, out_features=2),
            nn.BatchNorm1d(2),
            nn.Softmax(1)
        )

    def forward(self, data):
        data= self.layers(data)
        return data