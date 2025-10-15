import numpy as np


################### Class Components #################################################
# A:    type: Matrix of N x C_in x W_in;  data input 
# Z:    type: Matrix of N x C_in x W_in;  features after flatten
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x (C_in x W_in);  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x W_in;     how changes in inputs affect loss
######################################################################################
class Flatten:

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A = A
        Z = A.reshape(A.shape[0], -1)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        dLdA = dLdZ.reshape(self.A.shape)
        return dLdA
