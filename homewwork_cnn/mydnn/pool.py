import numpy as np
from resampling import *

################### Class Components #################################################
# kernel size:  K;  type scalar;    kernel size
# stride:           type: scalar;   stride
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input 
# Z:    type: Matrix of N x C_in x H_out x W_out;  features after pooling
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_in x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
######################################################################################

class MaxPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel
        self.max_indexes = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape        
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1  

        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        self.max_indexes = np.zeros((batch_size, in_channels, output_height, output_width, 2))
        for m in range(batch_size):            
            for c_in in range(in_channels):                    
                for h in range(0, output_height):            
                    for w in range(0, output_width):                                        
                        Z[m, c_in, h, w] = np.max(A[m, c_in, h:h + self.kernel, w:w + self.kernel])
                        block = A[m, c_in, h:h + self.kernel, w:w + self.kernel]
                        max_index = np.array(np.unravel_index(np.argmax(block), block.shape))
                        
                        self.max_indexes[m, c_in, h, w] = [max_index[0]+h, max_index[1] + w]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        for m in range(batch_size):
            for c_in in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        max_index = self.max_indexes[m, c_in, h, w].astype(np.int64)
                        dLdA[m, c_in, max_index[0], max_index[1]] += dLdZ[m, c_in, h, w]

        return dLdA


class MeanPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape        
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1  

        Z = np.zeros((batch_size, in_channels, output_height, output_width))        
        for m in range(batch_size):            
            for c_in in range(in_channels):                    
                for h in range(0, output_height):            
                    for w in range(0, output_width):                                        
                        Z[m, c_in, h, w] = np.mean(A[m, c_in, h:h + self.kernel, w:w + self.kernel])                        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        filter = np.full((self.kernel, self.kernel), 1/(self.kernel**2))
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        for m in range(batch_size):
            for c_in in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width):                                                
                        dLdA[m, c_in, h : h + self.kernel, w : w + self.kernel] += dLdZ[m, c_in, h, w] * 1/(self.kernel**2)                        
        return dLdA


class MaxPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A) # TODO: use maxpool2d_stride1.forward
        Z = self.downsample2d.forward(Z) # TODO: downsample2d.forward

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ) # TODO: use downsample2d.backward()
        dLdA = self.maxpool2d_stride1.backward(dLdZ) # TODO: use maxpool2d_stride1.backward()

        return dLdA


class MeanPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MeanPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A) #TODO use meanpool2d_stride1.forward()
        Z = self.downsample2d.forward(Z) #TODO use downsample2d.forward()

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ) #TODO use downsample2d.backward()
        dLdA = self.meanpool2d_stride1.backward(dLdZ) #TODO use meanpool2d_stride1.backward()

        return dLdA
