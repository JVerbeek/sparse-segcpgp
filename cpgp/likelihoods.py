# Taken from https://github.com/GPflow/GPflow/pull/1656/commits/704b9786a1bc54aba96d0ecaf472830156b0b90ehttps://github.com/GPflow/GPflow/pull/1656/commits/704b9786a1bc54aba96d0ecaf472830156b0b90e
from gpflow.likelihoods import ScalarLikelihood
from gpflow.base import Parameter
import tensorflow_probability as tfp
import tensorflow as tf
import gpflow

class NegativeBinomial(ScalarLikelihood):
    """
    The negative-binomial likelihood with pmf:
    .. math::
        NB(y \mid \mu, \psi) =
            \frac{\Gamma(y + \psi)}{y! \Gamma(\psi)}
            \left( \frac{\mu}{\mu + https://github.com/GPflow/GPflow/pull/1656/commits/704b9786a1bc54aba96d0ecaf472830156b0b90e\psi} \right)^y
            \left( \frac{\psi}{\mu + \psi} \right)^\psi
    where :math:`\mu = \exp(\nu)`. Its expected value is :math:`\mathbb{E}[y] = \mu `
    and variance :math:`Var[Y] = \mu + \frac{\mu^2}{\psi}`.
    """

    def __init__(self, psi=1.0, **kwargs):
        super().__init__(**kwargs)
        self.invlink = tf.exp
        self.psi = Parameter(
            psi,
            transform=tfp.bijectors.positive(lower=0.01)
        )

    def _scalar_log_prob(self, F, Y):
        mu = self.invlink(F)
        mu_psi = mu + self.psi
        psi_y = self.psi + Y
        f1 = (
                tf.math.lgamma(psi_y) -
                tf.math.lgamma(Y + 1.0) -
                tf.math.lgamma(self.psi)
        )
        f2 = Y * tf.math.log(mu / mu_psi)
        f3 = self.psi * tf.math.log(self.psi / mu_psi)
        return f1 + f2 + f3

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        mu = self.invlink(F)
        return mu + tf.pow(mu, 2) / self.psi


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    """
    The zero-inflated negative binomial distribution with pmf:
    .. math::
        ZINB(y \mid \mu, \psi, \theta) =
            \theta * I(y == 0) + (1 - \theta) NB(y \mid \mu, \psi)
    with expected value :math:`\mathbb{E}[y] = (1 - \theta)  \mu `
    and variance :math:`Var[Y] = (1 - \theta) \mu (1 + \theta * \mu + \mu / \psi )`
    """

    def __init__(self, theta=0.5, psi=1.0, **kwargs):
        super().__init__(psi, **kwargs)
        self.theta = Parameter(
            theta,
            transform=tfp.bijectors.positive(lower=0.01)
        )

    def _scalar_log_prob(self, F, Y):
        yz = tf.cast(Y == 0.0, dtype=gpflow.default_float())
        log_sup = super()._scalar_log_prob(F, Y)
        lse = yz * self.theta + (1.0 - self.theta) * tf.math.exp(log_sup)
        return tf.math.log(lse)

    def _conditional_mean(self, F):
        return (1.0 - self.theta) * self.invlink(F)

    def _conditional_variance(self, F):
        mu = self.invlink(F)
        return (1.0 - self.theta) * mu * (1.0 + self.theta * mu + mu / self.psi)