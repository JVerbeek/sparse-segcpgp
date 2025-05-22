import gpflow
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TypeVar, Union
from gpflow import set_trainable
from gpflow.base import Parameter
from gpflow.ci_utils import reduce_in_tests   # Cache stuff that does not depend on test points
f64 = gpflow.utilities.to_default_float

def select_trainable_parameters_without_prior(model: tf.Module) -> Dict[str, Parameter]:
    """Collects parameters with prior into a dictionary. (adapted from gpflow.utilities)"""
    return {
        k: p
        for k, p in gpflow.utilities.parameter_dict(model).items()
        if hasattr(p, "prior") and p.trainable and not p.prior
    }

def assign_prior(model, prior_scale=10, verbose=False):
    """Set Normal(loc, scale) priors for everything that needs a prior before starting HMC. This is a very lazy solution
    and should probably be extended with some param.param: tfp.distribution.Distribution dictionary structure to ensure that custom priors can be set.

    If attribute is Kernel or Likelihood, call function again. 
    If attribute is Parameter && trainable, set normal prior based on scaled fitted values and return
    Else pass.

    Arguments:
        model -- gpflow model

    Keyword Arguments:
        prior_scale -- scale of the normal prior (default: {10})
        verbose --  print what prior was assigned (default: {False})
    """
    for k, param in select_trainable_parameters_without_prior(model).items():
        if isinstance(param, gpflow.Parameter):   # If it is a trainable parameter we want it to have some vaguely reasonable normal distribution
            if "location" in k:
                param.prior = tfp.distributions.Uniform(low=f64(0), high=f64(400))
            else:
                loc = np.asarray(param).flatten()[0]
                scale = np.abs(loc*prior_scale) # It's variance and it being negative is not caught in HMC
                param.prior = tfp.distributions.Normal(loc=f64(loc), scale=f64(scale))
            if verbose:
                print(f"Assigned {param.prior} prior to {k}")
    return 


def assign_prior_informed(model, X, y, prior_scale=10, verbose=False):
    """Set Normal(loc, scale) priors for everything that needs a prior before starting HMC. This is a very lazy solution
    and should probably be extended with some param.param: tfp.distribution.Distribution dictionary structure to ensure that custom priors can be set.

    If attribute is Kernel or Likelihood, call function again. 
    If attribute is Parameter && trainable, set normal prior based on scaled fitted values and return
    Else pass.

    Arguments:
        model -- Model to set a prior for. Should be a gpflow.models.Model.
        X -- index
        y -- observations
    Keyword Arguments:
        prior_scale -- scale of the Normal prior (default: {10})
        verbose -- print what prior was assigned (default: {False})
    """
    for k, param in select_trainable_parameters_without_prior(model).items():
        if isinstance(param, gpflow.Parameter):   # If it is a trainable parameter we want it to have some vaguely reasonable normal distribution
            loc = np.asarray(param).flatten()[0]
            if "lengthscale" in k:
                scale = len(X)*2
            elif "location" in k:
                scale = len(X) / 5
            elif "steepness" in k:
                scale = 5*np.abs(y.max() - y.min()) / len(X)
            else:
                scale = np.abs(loc*prior_scale) # It's variance and it being negative is not caught in HMC
            param.prior = tfp.distributions.Normal(loc=f64(loc), scale=f64(scale))
            # else:
            #     param.prior = tfp.distributions.Dirichlet(loc)
            if verbose:
                print(f"Assigned {param.prior} to {k}")
    return 

def run_mala(model, step_size=0.1, num_burnin_steps=1000, num_samples=10000):
    """Run Metropolis-adjusted Langevin algorithm.

    Arguments:
        model -- gpflow model to do the sampling for.   

    Keyword Arguments:
        step_size -- step size of the sampler (default: {0.1})
        num_burnin_steps -- number of burnin steps to use (default: {1000})
        num_samples -- number of samples to take after burnin (default: {10000})

    Returns:
        model, samples, parameter_samples, sampling_helper -- model, samples, unconstrained samples, gpflow sampling helper
    """
    sampling_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    mala = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=sampling_helper.target_log_prob_fn,
        step_size=step_size,
        volatility_fn=None)

    @tf.function(reduce_retracing=True)
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=sampling_helper.current_state,
        kernel=mala,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=1,
        trace_fn=None,
        seed=42)
    
    samples = run_chain_fn()
    parameter_samples = sampling_helper.convert_to_constrained_values(samples)
    
    return model, samples, parameter_samples, sampling_helper

def run_hmc(model, leapfrog=10, step_size=1, num_burnin_steps=1000, num_samples=10000):
    """Run Hamiltonian MCMC.

    Arguments:
        model -- gpflow model to do the sampling for.   

    Keyword Arguments:
        leapfrog -- number of leapfrog steps (default: {1}))
        step_size -- step size of the sampler (default: {0.1})
        num_burnin_steps -- number of burnin steps to use (default: {1000})
        num_samples -- number of samples to take after burnin (default: {10000})

    Returns:
        model, samples, parameter_samples, sampling_helper -- model, samples, unconstrained samples, gpflow sampling helper
    """
    num_burnin_steps = reduce_in_tests(num_burnin_steps)
    num_samples = reduce_in_tests(num_samples)

    # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=leapfrog,
        step_size=step_size,
    )

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps=10,
        target_accept_prob=f64(0.75),
        adaptation_rate=0.1,
    )

    @tf.function(reduce_retracing=True)
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )


    samples, _ = run_chain_fn()
    parameter_samples = hmc_helper.convert_to_constrained_values(samples)

    return model, samples, parameter_samples, hmc_helper


def reversible_mcmc():
    
    # For iteration t > 1 
    # Perform within-model update
    sampling_helper = gpflow.optimizers.SamplingHelper(
    model.log_posterior_density, model.trainable_parameters
    )

    mala = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=sampling_helper.target_log_prob_fn,
        step_size=step_size,
        volatility_fn=None)

    # Perform between model update
    @tf.function(reduce_retracing=True)
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=sampling_helper.current_state,
        kernel=mala,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=1,
        trace_fn=None,
        seed=42)
    
    samples = run_chain_fn()
    parameter_samples = sampling_helper.convert_to_constrained_values(samples)
    
    return model, samples, parameter_samples, sampling_helper

