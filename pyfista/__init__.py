__author__ = "Martijn van den Ende"
__version__ = "1.0"

import numpy as np
from scipy.linalg import convolution_matrix

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from jax.experimental import host_callback

from time import time

from tqdm.notebook import tqdm

# tqdm progress bar snippet from Jeremie Coullon:
# https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/
def progress_bar_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
            message = f"Running for {num_samples:,} iterations"
    tqdm_bars = {}

    if num_samples > 100:
        print_rate = int(num_samples / 100)
    else:
        print_rate = 1 # if you run the sampler for less than 20 iterations
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples-remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return lax.cond(
            iter_num == num_samples-1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )


    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x   
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan


class FISTA:
    
    
    def __init__(self, lam, N, kernel=None, verbose=False):
        
        self.lam = lam
        self.N = N
        self.verbose = verbose
        if kernel is not None:
            self.update_kernel(kernel)
        
        pass

    def update_kernel(self, kernel):
        # Construct the minimal convolution kernel
        A = convolution_matrix(kernel, n=len(kernel), mode="same")
        # Get the eigenvalues
        w = jsp.linalg.eigh(A @ A, eigvals_only=True)
        # Construct the full size convolution kernel
        A = convolution_matrix(kernel, n=self.N, mode="same")
        self.kernel = jnp.array(A)
        # Define parameters (scaled by largest eigenvalue)
        self.L = 2 * w.max()
        self.rho = 1 / self.L
        self.lam2 = self.lam / self.L

        if self.verbose:
            print(f"Step size: {self.rho:.3e}")
            print(f"Sparsity constant: {self.lam:.3e}")
            print(f"Scaled sparsity constant: {self.lam2:.3e}")

        pass
    
    
    def solve(self, y, N):

        assert self.kernel is not None, "Kernel not initialised"
        
        # Initialise FISTA variables
        key = jax.random.PRNGKey(int(time()))
        x = jax.random.normal(key, shape=y.shape) / y.shape[1]
        r = x.copy()
        t = jnp.ones(r.shape[0])

        kernel = self.kernel
        rho = self.rho
        lam = self.lam2
        
        @jax.jit
        def soft(x, threshold):
            """Soft thresholding operation"""
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.)
        
        @jax.jit
        def fista_step(t, x, r, y):
            """Perform one FISTA step (for a single channel)"""
            y_hat = kernel @ x
            loss = jnp.linalg.norm(y - y_hat)
            x_new = soft(r - rho * kernel.T @ (kernel @ r - y), lam)
            t_new = 0.5 * (1 + jnp.sqrt(1 + 4 * t**2))
            t_ratio = ((t - 1) / t_new)
            r_new = x_new + t_ratio * (x_new - x)
            return t_new, x_new, r_new, loss
        
        # Parallelise fista_step
        do_step = jax.vmap(fista_step, in_axes=0, out_axes=0)

        # Main computation loop
        @progress_bar_scan(N)
        def body_fn(carry, i):
            t, x, r = carry
            t_new, x_new, r_new, loss = do_step(t, x, r, y)
            return (t_new, x_new, r_new), loss.mean()

        carry = (t, x, r)
        carry, loss = lax.scan(body_fn, carry, jnp.arange(N))
        # Final impulse model
        x = carry[1]
        # Reconstruction
        y_hat = np.array(x @ kernel)
        x = np.array(x)

        del r
        del t
        del carry
        
        return loss, x, y_hat

