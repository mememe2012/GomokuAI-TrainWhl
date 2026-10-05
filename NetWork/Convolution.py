import numpy as np
from rich import console

CONSOLE = console.Console()

class ConvOpe:
    """
    Convolution operation for 2D
    """
    def __init__(self, arrays:np.array, kernels:np.array):
        self.kernels = kernels
        self.arrays = arrays
    def opeOne(self, array:np.array, kernel:np.array):
        """
        Convolution operation for one kernel
        """
        array = self.padding(array, kernel.shape[0] // 2)
        result = np.zeros((array.shape[0]-kernel.shape[0]+1, array.shape[1]-kernel.shape[1]+1))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] = np.sum(array[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
        return result

    def padding(self, array:np.array, padding:int):
        """
        Padding operation
        - side = padding
        """
        return np.pad(array, ((padding, padding), (padding, padding)), 'constant')

    def operation(self):
        """
        Convolution operation for all kernels
        """
        result = []
        for array in self.arrays:
            for kernel in self.kernels:
                result.append(self.opeOne(array, kernel))
        return result

def view1D(array:np.array):
    length = 1
    for i in array.shape:
        length *= i
    return array.reshape(length)

def pooling(arrays:np.array, pool_size:int = 2, mode:str = 'average'):
        """
        Pooling operation
        - mode = average or max
        - every pool = (pool_size, pool_size)
        """
        result = []
        for array in arrays:
            if array.shape[0] % pool_size != 0:
                pad_size = pool_size - array.shape[0] % pool_size
                array = np.pad(array, ((0, pad_size), (0, pad_size)), 'constant')

            if mode == 'average':
                temp = np.zeros((array.shape[0]//pool_size, array.shape[1]//pool_size))
                for i in range(array.shape[0]//pool_size):
                    for j in range(array.shape[1]//pool_size):
                        temp[i, j] = np.average(array[i*pool_size:i*pool_size+pool_size, j*pool_size:j*pool_size+pool_size])
                result.append(temp)
            elif mode == 'max':
                temp = np.zeros((array.shape[0]//pool_size, array.shape[1]//pool_size))
                for i in range(array.shape[0]//pool_size):
                    for j in range(array.shape[1]//pool_size):
                        temp[i, j] = np.max(array[i*pool_size:i*pool_size+pool_size, j*pool_size:j*pool_size+pool_size])
                result.append(temp)
            else:
                CONSOLE.print(f'Error: Invalid mode\n- mode must be "average" or "max", "{mode}" given.', style='bold red')
                return
            
        result = np.array(result)
        return result