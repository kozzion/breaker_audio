from breaker_audio.filter_base import FilterBase

import numpy as np

from scipy.linalg import lstsq

class FilterConvolutionLinear(FilterBase):
    
    def __init__(self, size_kernel) -> None:
        self.size_kernel = size_kernel
        self.kernel = None

    def fit_transform(self, array_input, array_output_true):
        self._check_input(array_input, array_output_true)
        size_signal = array_input.shape[0]

        matrix_shifted = np.zeros((size_signal, self.size_kernel))
        for i in range(self.size_kernel):
            if i == 0: 
                matrix_shifted[i:,i] = array_input
            else:
                matrix_shifted[i:,i] = array_input[:-i]        
        self.kernel, res, rnk, s = lstsq(matrix_shifted, array_output_true) #TODO use gmres to get a smoother solution

        array_output_pred = self.transform(array_input)
        error = res

        return array_output_pred, error


    def transform(self, array_input):
        return np.convolve(array_input, self.kernel, 'same')
    