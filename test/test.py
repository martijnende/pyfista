import unittest

import numpy as np
import scipy.signal as sp
import jax

from pyfista import FISTA, get_kernel_bandwidth

jax.config.update("jax_platform_name", "cpu")


class TestFISTA(unittest.TestCase):

    def setUp(self):
        self.Nch, self.Nt = 1, 300
        self.ind_max, self.ind_min = 100, 150
        self.model = np.zeros((self.Nch, self.Nt))
        self.model[:, self.ind_max] = 1.0
        self.model[:, self.ind_min] = -0.5


    def test_kernel_bandwidth(self):

        widths = np.linspace(1, 20, 1000)

        for a in np.linspace(5, 10, 10):
            wavelet = sp.ricker(self.Nt, a)
            data = sp.convolve(self.model, wavelet[None, :], mode="same")
            a_opt = get_kernel_bandwidth(data, widths)
            self.assertAlmostEqual(np.abs(a_opt - a) / a, 0.0, delta=0.05)
    
    
    def test_deconvolution(self):

        ISTA_solver = FISTA(lam=1e0, N=self.Nt, kernel=None, verbose=False)
        
        for a in np.linspace(5, 10, 10):
            impulse_response = sp.ricker(self.Nt, a)
            impulse_response /= impulse_response.max()

            signal = sp.convolve(self.model, impulse_response[None, :], mode="same")
            kernel = sp.ricker(int(20 * a), a)
            kernel /= kernel.max()

            ISTA_solver.update_kernel(kernel)
            _, x, y_hat = ISTA_solver.solve(signal, N=1000)

            norm = np.linalg.norm(y_hat - signal) / np.linalg.norm(signal)

            self.assertAlmostEqual(norm, 0.0, delta=0.5)

            ind_max2 = np.argmax(x)
            ind_min2 = np.argmin(x)

            self.assertAlmostEqual(ind_max2, self.ind_max, delta=1.0)
            self.assertAlmostEqual(ind_min2, self.ind_min, delta=1.0)


if __name__ == "__main__":
    unittest.main()
