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
from check_shapes import check_shape as cs
from check_shapes import check_shapes, inherit_check_shapes

from gpflow.base import Parameter, TensorType
from gpflow.utilities import positive
from gpflow.kernels.base import Combination, Kernel
import gpflow

# test mports
import matplotlib.pyplot as plt

class ChangePoints2D(Combination):
    r"""
    The ChangePointsND kernel defines a fixed number of change boundaries along a 2d
    input space where different kernels govern different parts of the space.

    The kernel is by multiplication and addition of the base kernels with
    sigmoid functions (σ). A single change-point kernel is defined as::

        K₁(x, x') * (1 - σ(x)) * (1 - σ(x')) + K₂(x, x') * σ(x) * σ(x')

    where K₁ is deactivated around the change-point and K₂ is activated. The
    single change-point version can be found in :cite:t:`lloyd2014`. Each sigmoid
    is a logistic function defined as::

        σ(x) = 1 / (1 + exp{-s(x - x₀)})

    parameterized by location "x₀" and steepness "s".

    The key reference is :cite:t:`lloyd2014`.
    """

    def __init__(
        self,
        kernels: Sequence[Kernel],
        boundaries: TensorType,
        steepness: TensorType = 1.0,
        name: Optional[str] = None,
    ):
        """
        :param kernels: list of kernels defining the different regimes
        :param boundaries: list of change-point locations in the 1d input space
        :param steepness: the steepness parameter(s) of the sigmoids, this can be
            common between them or decoupled
        """
        # if len(kernels) != len(boundaries) + 1:
        #     raise ValueError(
        #         "Number of kernels ({nk}) must be one more than the number of "
        #         "changepoint locations ({nl})".format(nk=len(kernels), nl=len(locations))
        #     )

        if isinstance(steepness, Sequence) and len(steepness) != len(locations):
            raise ValueError(
                "Dimension of steepness ({ns}) does not match number of changepoint "
                "locations ({nl})".format(ns=len(steepness), nl=len(locations))
            )

        super().__init__(kernels, name=name)
        self.boundaries = boundaries  # should be a list of functions. Needs some sort of kernel id to associate them with a kernel. Now assume that they are in same order as kernels
       
        # self.locations = Parameter(locations)  # should be the result of a boundary function.
        self.steepness = Parameter(steepness, transform=positive())   # I think this should be a vector
        self.information = []

    def _set_kernels(self, kernels: Sequence[Kernel]) -> None:
        # it is not clear how to flatten out nested change-points
        self.kernels = list(kernels)

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        #cs(X, "[batch..., N, 1]  # The `ChangePoints` kernel requires a 1D input space.")

        rank = tf.rank(X) - 2
        ndim = tf.rank(X)
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        Ncp = tf.constant(len(self.boundaries))
        sig_X = self._sigmoids(X)

        if X2 is None:
            rank2 = 0
            batch2 = tf.constant([], dtype=tf.int32)
            N2 = N
            sig_X2 = sig_X
            sig_X = tf.reshape(sig_X, [N, 1, Ncp], 0) # batch, 1, N N, Ncp
            sig_X2 = tf.reshape(sig_X2, [1, N, Ncp])   # batch, N, N, 1.... Ncp
          
        else:
            rank2 = tf.rank(X2) - 2
            batch2 = tf.shape(X2)[:-2]
            N2 = tf.shape(X2)[-2]

            sig_X2 = self._sigmoids(X2)
            ones = tf.ones((rank,), dtype=tf.int32)
            ones2 = tf.ones((rank2,), dtype=tf.int32)
            sig_X = tf.reshape(sig_X, tf.concat([[N], ones2, [1, Ncp]], 0))
            sig_X2 = tf.reshape(sig_X2, tf.concat([ones, [1], [N2 ], [Ncp]], 0))

        starters = sig_X * sig_X2
        stoppers = (1 - sig_X) * (1 - sig_X2)

        # `starters` are the sigmoids going from 0 -> 1, whilst stoppers go from 1 -> 0
        # prepend `starters` with ones and append ones to `stoppers` since the
        # first kernel has no start and the last kernel has no end
        ones = tf.ones([N, N2, 1], dtype=X.dtype)    
        starters = tf.concat([ones, starters], axis=-1)
        stoppers = tf.concat([stoppers, ones], axis=-1)

        # now combine with the underlying kernels
        kernel_stack = tf.stack([k(X, X2) for k in self.kernels], axis=-1)
        return tf.reduce_sum(kernel_stack * starters * stoppers, axis=-1)

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        #@TODO re-implement shape checking
        batch = tf.shape(X)[:-2] 
        
        N = tf.shape(X)[-2]
        Ncp = len(self.boundaries)

        sig_X = tf.reshape(self._sigmoids(X), (N, Ncp))   #@TODO fix first argument of concat to include [N for _ in range(rank)]

        ones = tf.ones(tf.concat([[N, 1]], 0), dtype=X.dtype)   #@TODO fix first argument of concat to include [N for _ in range(rank)]
        starters = tf.concat([ones, sig_X * sig_X], axis=-1)
        stoppers = tf.concat([(1 - sig_X) * (1 - sig_X), ones], axis=-1)

        kernel_stack = tf.stack([k(X, full_cov=False) for k in self.kernels], axis=-1)
        return tf.reduce_sum(kernel_stack * starters * stoppers, axis=-1)

    def _sigmoids(self, X: tf.Tensor) -> tf.Tensor:
        X, Y = X[...,0], X[...,1]
        X_prime = tf.cast((X + Y), tf.float32)
        boundary = self.boundaries[0](X, Y) # assume boundary is a class implementing __call__
        steepness = tf.reshape(self.steepness, (-1,))
        return tf.sigmoid(-steepness * tf.cast((X_prime[..., None] - boundary), tf.double))  # broadcasting; resulting shape is batch, rest, Ncp.


class PolyBoundary(gpflow.Module):
    def __init__(self, degree, weights):
        self.w = Parameter(tf.expand_dims(tf.constant(weights[:-1], dtype=tf.float32), -1))
        self.b = Parameter(tf.reshape(tf.constant(weights[-1], dtype=tf.float32), (-1, 1)))
        self.deg = degree

    def __call__(self, X: tf.Tensor, Y: tf.Tensor):
        Y = tf.expand_dims(Y, -1)
        if self.deg > 2:
            X_ = tf.stack([X for _ in range(self.deg-1)], -1)
            Xb = tf.concat([X_, Y, tf.ones(Y.shape, dtype=tf.double)], -1)
            pows = (tf.concat((tf.range(self.deg, 0, -1, dtype=tf.double), tf.ones((1,), dtype=tf.double)), axis=0))
            Xbf = tf.pow(Xb, pows)
            wb = tf.concat((self.w, self.b), 0)
        elif self.deg == 2:
            X_ = tf.expand_dims(X, -1)
            Xb = tf.concat([X_, Y, tf.ones(Y.shape, dtype=tf.double)], -1)
            pows = (tf.concat((tf.range(self.deg, 0, -1, dtype=tf.double), tf.ones((1,), dtype=tf.double)), axis=0))
            Xbf = tf.pow(Xb, pows)
            wb = tf.concat((self.w, self.b), 0)
        else:  # its linear
            X_ = tf.expand_dims(X, -1)
            Xbf = tf.concat([X_, Y, tf.ones(Y.shape, dtype=tf.double)], -1)
            wb = tf.concat((self.w, self.w, self.b), 0)
        return tf.cast((Xbf @ wb), tf.float32)


class LinearBoundary(gpflow.Module):
    def __init__(self):
        self.w = Parameter(tf.expand_dims(tf.constant([1], dtype=tf.double), -1)) #w2, w1, b
    
    def __call__(self, X: tf.Tensor, Y: tf.Tensor):
        X_ = tf.expand_dims(X, -1)
        Y = tf.expand_dims(Y, -1)
        Xb = tf.concat([X_.T, Y, tf.ones((1, 1), dtype=tf.double)], -1)
        return tf.cast((Xb @ self.w), tf.float32)

if __name__ == "__main__":
    xx = tf.linspace(-10, 10, 100) 
    yy = tf.linspace(-10, 10, 100)
    X, Y = tf.meshgrid(xx, yy)
    poly = PolyBoundary(3, [0, 1, 1, 1])
    boundaries = [poly]
    kernels = [gpflow.kernels.RBF(), gpflow.kernels.Constant()]
    steepness = 0.5
    K = ChangePoints2D(kernels, boundaries, steepness)

    X = tf.reshape(tf.concat([tf.squeeze(xx)[None], tf.squeeze(yy)[None]], 0), (-1, 2))
    Xp = tf.reshape(tf.concat([tf.squeeze(xx)[None], tf.squeeze(yy)[None]], 0), (-1, 2))
    K(X, Xp)