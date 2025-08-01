import os
import numpy as np
import fitsio as fio
import galsim.roman as roman
from . import RomanEffects
from .utils import sca_number_to_file


class IPC(RomanEffects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)

        self.is_model_valid()

    def simple_model(self, image):
        self.logger.warning("Simple model will be applied for IPC effect.")
        kernel = roman.ipc_kernel
        image.applyIPC(kernel, edge_treatment="extend", fill_value=None)
        return image

    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No IPC kernel data file provided; no IPC effect will be applied.")
            return image
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))

        self.logger.warning("Lab measured model will be applied for IPC effect.")
        # pad the array by one pixel at the four edges
        num_grids = 4  # num_grids <= 8
        grid_size = 4096 // num_grids

        array_pad = image.array[4:-4, 4:-4]  # it's an array instead of img
        array_pad = np.pad(array_pad, [(5, 5), (5, 5)], mode="symmetric")  # 4098x4098 array

        K = self.df["IPC"][:, :, :, :]  # 3,3,512, 512

        t = np.zeros((grid_size, 512))
        for row in range(t.shape[0]):
            t[row, row // (grid_size // 512)] = 1

        array_out = np.zeros((4096, 4096))
        # split job in sub_grids to reduce memory
        for gj in range(num_grids):
            for gi in range(num_grids):
                K_pad = np.zeros((3, 3, grid_size + 2, grid_size + 2))

                for j in range(3):
                    for i in range(3):
                        tmp = (t.dot(K[j, i, :, :])).dot(t.T)  # grid_sizexgrid_size
                        K_pad[j, i, :, :] = np.pad(tmp, [(1, 1), (1, 1)], mode="symmetric")

                for dy in range(-1, 2):
                    for dx in range(-1, 2):

                        array_out[
                            gj * grid_size : (gj + 1) * grid_size, gi * grid_size : (gi + 1) * grid_size
                        ] += (
                            K_pad[1 + dy, 1 + dx, 1 - dy : 1 - dy + grid_size, 1 - dx : 1 - dx + grid_size]
                            * array_pad[
                                1 - dy + gj * grid_size : 1 - dy + (gj + 1) * grid_size,
                                1 - dx + gi * grid_size : 1 - dx + (gi + 1) * grid_size,
                            ]
                        )

        image.array[:, :] = array_out
        return image
