import numpy as np
from resampling import *
from Conv1d import *
from Conv2d import *

################### ConvTranspose1d Class Components ###################################
# upsampling_factor:    type: scalar;       upsampling factor \in Z^+ 
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x W_in;    data input 
# Z:    type: Matrix of N x C_out x W_out;  features after ConvTranspose1d
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x W_in;    how changes in inputs affect loss
###################################################################################
class ConvTranspose1d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        upsampling_factor,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        self.upsample1d = Upsample1d(upsampling_factor)
        self.conv1d_stride1 = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            weight_init_fn=weight_init_fn,
            bias_init_fn=bias_init_fn,
        )

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        upsample = self.upsample1d.forward(A)

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(upsample)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call backward in the correct order
        conv = self.conv1d_stride1.backward(dLdZ)

        # Upsampling1d backward
        dLdA = self.upsample1d.backward(conv)

        return dLdA


################### ConvTranspose2d Class Components ###################################
# upsampling_factor:    type: scalar;       upsampling factor \in Z^+ 
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input 
# Z:    type: Matrix of N x C_out x H_out x W_out;  features after ConvTranspose2d
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
######################################################################################
class ConvTranspose2d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        upsampling_factor,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )                
        self.upsample2d = Upsample2d(self.upsampling_factor)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        upsample = self.upsample2d.forward(A)

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(upsample)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call backward in correct order
        # Upsampling2d backward
        conv = self.conv2d_stride1.backward(dLdZ)

        dLdA = self.upsample2d.backward(conv)

        return dLdA
