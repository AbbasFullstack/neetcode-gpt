import numpy as np
from numpy.typing import NDArray

class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Numerical stability ke liye max subtract karo
        z_shifted = z - np.max(z)
        
        # Exponential nikaalo
        exp_z = np.exp(z_shifted)
        
        # Sum se divide karo
        result = exp_z / np.sum(exp_z)
        
        # 4 decimal places tak round karo
        return np.round(result, 4)