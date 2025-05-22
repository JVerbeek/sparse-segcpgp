# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence

import tensorflow as tf
from gpflow.base import Parameter, TensorType
from gpflow.utilities.ops import square_distance, difference_matrix

from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm

import numpy as np
import tensorflow_probability as tfp
import gpflow
f64 = gpflow.utilities.to_default_float
from gpflow.kernels import Kernel, Sum
import pandas as pd
import os
from scipy.integrate import cumtrapz
import scipy.signal as signal
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def get_kernel_sample(xx, n_kernel_samples, k, mu=0):
    """Utility function, return sample from kernel k over range x_min, x_max"""
    return xx, np.random.multivariate_normal(np.ones(len(xx)) * mu, k(xx), n_kernel_samples).T

def create_dataset(xx, n_kernel_samples, k, filename="dataset"):
    """Create .csv dataset containing X and n_kernel_samples samples from kernel k.
    Possibly change to .json so that breakpoint can also be given.
    """
    _, yy = get_kernel_sample(xx, n_kernel_samples, k)
    df1 = pd.DataFrame(xx)
    df2 = pd.DataFrame(yy)
    df = pd.concat([df1, df2], axis=1)
    header = ["x"] + [f"y{i}" for i in range(yy.shape[1])]
    df.to_csv(f"{os.getcwd()}/data/{filename}.csv", header=header)
    return

class SpectralMixture(Kernel):
    """Implements a spectral mixture kernel as described by Wilson and Adams."""
    def __init__(self, Q, mixture_weights=None, frequencies=None, lengthscales=None, max_freq=1.0, max_length=1.0, x=None, y=None, fs=1, active_dims: slice | Sequence[int] | None = None, name: str | None = None) -> None:
        super().__init__(active_dims, name)
        self.Q = Q
        self.logging = []
        if y is not None:
            frequencies, lengthscales, mixture_weights = self.initialize_from_emp_spec(Q, x, y, fs)
        else:
            if mixture_weights is None:
                mixture_weights = [1/Q for i in range(Q)]
            if frequencies is None:
                frequencies = [((i + 1) / Q) * max_freq for i in range(Q)]
            if lengthscales is None:
                lengthscales = [max_length for _ in range(Q)]   


        self.logging.append((frequencies, lengthscales, mixture_weights))
        kernels = [SpectralMixtureComponent(i + 1, mixture_weights[i], frequencies[i], lengthscales[i], active_dims=active_dims) for i in range(len(frequencies))]
        self._kernel = Sum(kernels)
    

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        return self._kernel.K(X, X2)
    
    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self._kernel.K_diag(X)

    def plot(self, inv_spec, means, varz):
        """Plot LS periodogram, Gaussians."""
        plt.hist(inv_spec, bins=100, color='blue')
        for m, s in zip(means.reshape((-1,)), varz.reshape((-1,))):
            plt.plot(np.linspace(0, 1, 100), norm.pdf(np.linspace(0, 1, 100), m, np.sqrt(s)), color="orange")
        plt.title("Lomb-Scargle periodogram")
        plt.show()
    

    def initialize_from_emp_spec(self, Q, x, y, fs):
        """
        Initializes the Spectral Mixture hyperparameters by fitting a GMM on the empirical spectrum,
        found by Lomb-Scargle periodogram.
        Function largely taken from: https://docs.gpytorch.ai/en/v1.1.1/_modules/gpytorch/kernels/spectral_mixture_kernel.html#SpectralMixtureKernel.initialize_from_data_empspect
        Instead, here the Lomb-Scargle periodogram is used to fit the GMM to allow analysis of ununiformly sampled data.

        :param Q (int) number of spectral components in SM kernel
        :param x (np.array of float64) X values of input data
        :param y NumPy array of float64. Y values of input data

        return: frequencies lengthscales, mixture weights, all of which are NumPy arrays of shape (Q,)
        """

        freqs = np.linspace(0.01, fs, 1000)
        
        Pxx = signal.lombscargle(x.flatten(), y.flatten(), freqs=freqs, normalize=True)
        total_area = np.trapz(Pxx, freqs)
        spec_cdf = np.hstack((np.zeros(1), cumtrapz(Pxx, freqs)))
        spec_cdf = spec_cdf / total_area

        a = np.random.rand(10000, 1)
        _, q = np.histogram(a, spec_cdf, density=True)
        bins = np.digitize(a, q)
        slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freqs[bins] - freqs[bins - 1])
        intercepts = spec_cdf[bins - 1] - slopes * freqs[bins - 1]
        inv_spec = (a - intercepts) / slopes
        #GMM = GaussianMixture(n_components=q, init_params='k-means++')
        GMM = GaussianMixture(n_components=Q, init_params="random_from_data")
        GMM.fit(X=inv_spec)
        means = GMM.means_
        varz = GMM.covariances_
        weights = GMM.weights_
        emp_frequencies, emp_lengthscales, emp_mixture_weights = means.flatten(), varz.flatten(), weights.flatten()
        lengthscales = 1 / np.sqrt(emp_lengthscales)
        mixture_weights = emp_mixture_weights
        frequencies = emp_frequencies
        return frequencies, lengthscales, mixture_weights


class SpectralMixtureComponent(Kernel):
    """
    Single component of the SM kernel by Wilson-Adams (2013).
    k(x,x') = w * exp(-2 pi^2 * |x-x'| * sigma_q^2 ) * cos(2 pi |x-x'| * mu_q)
    """
    def __init__(self, index, mixture_weight, frequency, lengthscale, active_dims):
        super().__init__(active_dims=active_dims)
        self.index = index
        
        def logit_transform(min, max):
            a = tf.cast(min, tf.float64)
            b = tf.cast(max, tf.float64)
            shift = tfp.bijectors.Shift(a)
            scale = tfp.bijectors.Scale((b - a))
            sigmoid = tfp.bijectors.Sigmoid()
            logistic = tfp.bijectors.Chain([shift, scale, sigmoid])
            return logistic
        logistic = tfp.bijectors.Sigmoid(low=tf.cast(0.00001, tf.float64), high=tf.cast(900000, tf.float64), validate_args=True) # numerical stability
        trainable = True
        self.mixture_weight = gpflow.Parameter(mixture_weight, transform=logistic, trainable=trainable)
        self.frequency = gpflow.Parameter(frequency, transform=logistic, trainable=trainable)
        self.lengthscale = gpflow.Parameter(lengthscale, transform=logistic, trainable=trainable)
        

    def K(self, X, X2=None):
        """Kernel function."""
        if X2 is None:
            X2 = X

        # tau_squared = tf.reduce_sum(self.scaled_difference_matrix(X, X2), axis=-1)
        tau_squared = self.scaled_squared_euclid_dist(X, X2)
        exp_term = tf.exp(-2.0 * (np.pi ** 2) * tau_squared)

        # Following lines are taken from Sami Remes' implementation (see references above)
        f = tf.expand_dims(X, 1)
        f2 = tf.expand_dims(X2, 0)
        freq = tf.expand_dims(self.frequency, 0)
        freq = tf.expand_dims(freq, 0)
        r = tf.reduce_sum(freq * (f - f2), 2)
        cos_term = tf.cos(r)
        return self.mixture_weight * exp_term * cos_term # * 2 * np.pi

    def scale(self, X):
        """Scale X by 1/lengthscale."""
        X_scaled = X / self.lengthscale if X is not None else X
        return X_scaled

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.mixture_weight))


    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Function to overwrite gpflow.kernels.stationaries
        Returns ||(X - X2ᵀ) / ℓ||² i.e. squared L2-norm.
        """
        return square_distance(self.scale(X), self.scale(X2))
    
    def scaled_difference_matrix(self, X, X2=None):
        return difference_matrix(self.scale(X), self.scale(X2))
