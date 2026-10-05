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
        arrays = np.asarray(self.arrays)
        kernels = np.asarray(self.kernels)
        if arrays.ndim not in (3, 4) or kernels.ndim != 3:
            raise ValueError(
                "Convolution expects arrays with shape (channels, height, width) "
                "or (batch, channels, height, width), and kernels with shape "
                "(kernels, height, width)."
            )

        kernel_height, kernel_width = kernels.shape[-2:]
        padded = np.pad(
            arrays,
            ((0, 0),) * (arrays.ndim - 2)
            + (
                (kernel_height // 2, kernel_height // 2),
                (kernel_width // 2, kernel_width // 2),
            ),
            mode="constant",
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (kernel_height, kernel_width), axis=(-2, -1)
        )

        if arrays.ndim == 3:
            result = np.einsum("nhwkl,ckl->nchw", windows, kernels, optimize=True)
            return result.reshape(-1, *result.shape[-2:])

        result = np.einsum("bnhwkl,ckl->bnchw", windows, kernels, optimize=True)
        return result.reshape(
            arrays.shape[0], -1, *result.shape[-2:]
        )

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
        if pool_size < 1:
            raise ValueError("pool_size must be at least 1.")
        if mode not in ("average", "max"):
            CONSOLE.print(f'Error: Invalid mode\n- mode must be "average" or "max", "{mode}" given.', style='bold red')
            return

        arrays = np.asarray(arrays)
        if arrays.ndim not in (3, 4):
            raise ValueError(
                "Pooling expects arrays with shape (channels, height, width) "
                "or (batch, channels, height, width)."
            )

        height, width = arrays.shape[-2:]
        pad_height = (-height) % pool_size
        pad_width = (-width) % pool_size
        padding = ((0, 0),) * (arrays.ndim - 2) + (
            (0, pad_height),
            (0, pad_width),
        )
        if pad_height or pad_width:
            arrays = np.pad(arrays, padding, mode="constant")

        pooled_shape = arrays.shape[:-2] + (
            arrays.shape[-2] // pool_size,
            pool_size,
            arrays.shape[-1] // pool_size,
            pool_size,
        )
        windows = arrays.reshape(pooled_shape)
        reducer = np.mean if mode == "average" else np.max
        return reducer(windows, axis=(-3, -1))