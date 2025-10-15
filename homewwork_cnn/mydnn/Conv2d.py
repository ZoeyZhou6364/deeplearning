import numpy as np
from resampling import *

################### Conv2d_stride1 and Conv2d Class Components ###################################
# kernel size:  K;  type scalar;    kernel size
# stride:           type: scalar;   downsampling factor
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input for convolution
# Z:    type: Matrix of N x C_out x H_out x W_out;  features after conv2d with stride 1
# ------------------------------------------------------------------------------------
# W:    type: Matrix of C_out x C_in X K X K;   weight parameters, i.e. kernels
# b:    type: Matrix of C_out x 1;              bias parameters
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
# dLdW: type: Matrix of C_out x C_in X K X K;       how changes in weights affect loss
# dLdb: type: Matrix of C_out x 1;                  how changes in bias affect loss
######################################################################################

class Conv2d_stride1:
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
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape                

        self.output_height = input_height - self.kernel_size + 1
        self.output_width = input_width - self.kernel_size + 1        

        Z = np.zeros((batch_size, self.out_channels, self.output_height, self.output_width))
        for m in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(in_channels):                    
                    for h in range(0, self.output_height):            
                        for w in range(0, self.output_width):            
                            for k1 in range(0, self.kernel_size):
                                for k2 in range(0, self.kernel_size):                                    
                                    Z[m, c_out, h, w] += A[m, c_in, h + k1, w + k2]*self.W[c_out, c_in, k1, k2]
                Z[m, c_out, :] += self.b[c_out]            
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        padded_dLdZ_width = output_width + (self.kernel_size - 1) * 2
        padded_dLdZ_height = output_height + (self.kernel_size - 1) * 2
        padded_dLdZ = np.zeros((batch_size, out_channels, padded_dLdZ_height, padded_dLdZ_width))
        padded_dLdZ[:, :, self.kernel_size - 1:self.kernel_size - 1 + output_height, self.kernel_size - 1:self.kernel_size - 1 + output_width] = dLdZ
        flipped_W = np.flip(np.flip(self.W, 2), 3)
        
        self.dLdW = np.zeros((out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.dLdb = np.zeros((out_channels))
        for c_out in range(out_channels):
            self.dLdb[c_out] += np.sum(dLdZ[:, c_out, :, :])
            for m in range(batch_size):            
                for c_in in range(self.in_channels):                        
                    for k1 in range(self.kernel_size):                    
                        for k2 in range(self.kernel_size):                    
                            for h in range(output_height):
                                for w in range(output_width):
                                    self.dLdW[c_out, c_in, k1, k2] += self.A[m, c_in, k1 + h, k2 + w]*dLdZ[m, c_out, h, w]
                
        batch_size, in_channels, input_height, input_width = self.A.shape
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        for m in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for h in range(input_height):
                        for w in range(input_width):
                            for k1 in range(self.kernel_size):
                                for k2 in range(self.kernel_size):
                                    dLdA[m, c_in, h, w] += padded_dLdZ[m, c_out, h + k1, w + k2]*flipped_W[c_out, c_in, k1, k2] 
        
        return dLdA


class Conv2d:
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
            

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        # if self.conv2d_stride1.in_channels != A.shape[1]:
        Z_pad = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)))        
        
        # Call Conv2d_stride1        
        conv = self.conv2d_stride1.forward(Z_pad)

        # downsample
        Z = self.downsample2d.forward(conv)        

        return Z

    def backward(self, dLdZ):
        # Calculate dLdA
        # Line 1: Downsample1d backward
        # Line 2: Conv1d backward
        # Line 3: Unpad
        # downSample = self.downsample1d.backward(dLdZ)
        # conv = self.conv1d_stride1.backward(downSample)        
        # dLdA = conv[:, :, self.pad : conv.shape[2] - self.pad]              
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad : dLdA.shape[2]-self.pad, self.pad : dLdA.shape[3]-self.pad]

        return dLdA
