import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, output_dim=None, input_size=None):
        """

        The Classifer class: We are developing a model similar to ACTINN for good accuracy

        """
        if output_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input dim (num features) and output dim (number of classes)')

        super(Classifier, self).__init__();
        self.inp_dim = input_size;
        self.out_dim = output_dim;

        # feed forward layers
        self.classifier_sequential = nn.Sequential(
            nn.Linear(self.inp_dim, 4000),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(4000, 1500),

            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(1500, 500),

            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(500, 200),

            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, output_dim),
            # SoftMax not needed for CrossEntropyLoss in PyTorch
            ## i.e. no need for nn.Softmax(dim=1)
        )

    def forward(self, x):
        """

        Forward pass of the classifier

        """

        out = self.classifier_sequential(x);

        return out
