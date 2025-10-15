# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

################### Conv1d_stride1 and Conv1d Class Components ###################################
# kernel size:  K;      type scalar;        kernel size
# stride:               type: scalar;       equivalent to downsampling factor
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x W_in;    data input for convolution
# Z:    type: Matrix of N x C_out x W_out;  features after conv1d with stride
# ------------------------------------------------------------------------------------
# W:    type: Matrix of C_out x C_in X K;   weight parameters, i.e. kernels
# b:    type: Matrix of C_out x 1;          bias parameters
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x W_in;    how changes in inputs affect loss
# dLdW: type: Matrix of C_out x C_in X K;   gradient of Loss w.r.t. weights
# dLdb: type: Matrix of C_out x 1;          gradient of Loss w.r.t. bias
###################################################################################
class Conv1d_stride1:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size))
        for m in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(in_channels):
                    for w in range(0, output_size):            
                        for k in range(0, self.kernel_size):
                            Z[m, c_out, w] += A[m, c_in, w + k]*self.W[c_out, c_in, k]
                Z[m, c_out, :] += self.b[c_out]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        padded_dLdZ_size = output_size + (self.kernel_size - 1) * 2
        padded_dLdZ = np.zeros((batch_size, out_channels, padded_dLdZ_size))
        padded_dLdZ[:, :, self.kernel_size - 1:self.kernel_size - 1+output_size] = dLdZ
        flipped_W = np.flip(self.W, 2)


        self.dLdb = np.zeros((out_channels))
        self.dLdW = np.zeros((out_channels, self.in_channels, self.kernel_size))

        for c_out in range(out_channels):
            self.dLdb[c_out] += np.sum(dLdZ[:, c_out, :])
            for m in range(batch_size):            
                for c_in in range(self.in_channels):                        
                    for k in range(self.kernel_size):                    
                        for w in range(output_size):                                                    
                            self.dLdW[c_out, c_in, k] += self.A[m, c_in, k + w]*dLdZ[m, c_out, w]
                
        batch_size, in_channels, input_size = self.A.shape
        dLdA = np.zeros((batch_size, in_channels, input_size))

        for m in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for w in range(input_size):
                        for k in range(self.kernel_size):
                            dLdA[m, c_in, w] += padded_dLdZ[m, c_out, w + k]*flipped_W[c_out, c_in, k] 

        return dLdA


class Conv1d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Calculate Z
        # Line 1: Pad with zeros
        # Line 2: Conv1d forward
        # Line 3: Downsample1d forward
        batch_size, in_channels, input_size = A.shape
        Z_pad = np.zeros((batch_size, in_channels, input_size + self.pad*2))
        Z_pad[:, :, self.pad:input_size+self.pad] = A
        conv = self.conv1d_stride1.forward(Z_pad)
        Z = self.downsample1d.forward(conv)        

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Calculate dLdA
        # Line 1: Downsample1d backward
        # Line 2: Conv1d backward
        # Line 3: Unpad
        downSample = self.downsample1d.backward(dLdZ)
        conv = self.conv1d_stride1.backward(downSample)        
        dLdA = conv[:, :, self.pad : conv.shape[2] - self.pad]        
        
        return dLdA
