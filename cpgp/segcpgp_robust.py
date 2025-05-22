import sys
import os
import gpflow as gpf
import numpy as np
np.random.seed(314)
f64 = gpf.utilities.to_default_float
sys.path.append(os.path.expanduser("~")+"/changepoint-gp/")
from cpgp.hmc import assign_prior, run_mala
from cpgp.spectral_mixture import SpectralMixture
from RCGP.rcgp.rcgp import RCGPR
from RCGP.rcgp.w import IMQ 
from scipy.stats.distributions import chi2
import tensorflow as tf

def get_kernel(name, X, y):
    """Get kernel based on name. X and y need to be passed for the spectral mixture GMM to fit on.

    Arguments:
        name -- name of the kernel to be returned.
        X -- index.
        y -- observations.

    Returns:
        k -- gpflow Kernel object. 
    """
    name = name.lower()
    if "noise" in name and "spectral" not in name:
        k = gpf.kernels.White(0.1)
        # k.variance = gpflow.Parameter(np.std(y), transform=tfp.bijectors.SoftClip(gpflow.utilities.to_default_float(0.1*np.std((y))), gpflow.utilities.to_default_float(100000*np.std(y))))
    if "spectral-" in name:
        q = int(name.split("-")[1])
        k = SpectralMixture(q, x=X, y=y)
        # k.lengthscale = gpflow.Parameter(len(X)/3, transform=tfp.bijectors.SoftClip(gpflow.utilities.to_default_float(len(X)/10), gpflow.utilities.to_default_float(3*len(X))))
        if "noise" in name:
            q = gpf.kernels.White(0.01)
            # q.variance = gpflow.Parameter(np.std(y), transform=tfp.bijectors.SoftClip(gpflow.utilities.to_default_float(.1*np.std(y)), gpflow.utilities.to_default_float(100000*np.std(y))))
            k += q
    if "linear" in name:
        k = gpf.kernels.Linear()  # + gpf.kernels.RBF()
    if "matern" in name:
        k = gpf.kernels.Matern52()  # + gpf.kernels.RBF()
    if "constant" in name:
        k = gpf.kernels.Constant(1)
        if "noise" in name:
            k += gpf.kernels.White(0.01)
    if "rbf" in name:
        k = gpf.kernels.RBF()
        if "noise" in name:
            k += gpf.kernels.White(0.01)
    if "per" in name:
        k = gpf.kernels.Periodic(gpf.kernels.RBF(), 1)

    return k


class SegCPGP():
    """Implements Segmenting Changepoint Gaussian Processes"""

    def __init__(self, pval=0.1, df=1, stepsize=0.1, burnin=1000, samples=5000, logging=False, sampling=False) -> None:
        self.LOCS = []
        self.STEEPNESS = []
        self.TESTED = []
        self.lrts = []
        self.logging = logging
        self.sampling = sampling
        self.pval = pval
        self.stepsize = stepsize
        self.burnin = burnin
        self.samples = samples
        self.df = df

    def fit(self, X, y, base_kernel_name="constant", custom_kernel=None):
        """Fit SegCPGP

        Arguments:
            X -- index
            y -- observations

        Keyword Arguments:
            base_kernel_name -- kernel used in the CP kernel (default: {"constant"})

        Returns:
            _description_
        """
        results = self.call(X, y, base_kernel_name, custom_kernel)
        return results

    def get_high_likelihood_model(self, X, y, model_name, base_kernel_name, custom_kernel, n_attempts=3, fit_noise=True):
        """The LML has no guarantees that it does not end up in a local optimum.
        We select the model with the highest likelihood after a number of attempts. 

        Arguments:
            attempts -- number of attempts to make

        Returns: 
            model -- model with the highest likelihood.
        """
        def get_model(X, y, model_name, base_kernel_name, custom_kernel):
            """Utility function for getting either a GPR or a CP based on the model_name string."""
            if not custom_kernel:
                kernels = [get_kernel(base_kernel_name, X, y), get_kernel(base_kernel_name, X, y)]
            else:
                kernels = [custom_kernel, custom_kernel]
            if model_name == "cp":
               # model = RCGPR((X, y), weighting_function=IMQ(C=np.quantile(np.abs(y), 0.7)), kernel=gpf.kernels.ChangePoints(kernels=kernels, locations=[np.random.randint(X.min(), X.max())], steepness=[100]))
                model = gpf.models.SVGP(likelihood=gpf.likelihoods.StudentT(), inducing_variable=np.linspace(0, max(X), 20), kernel=gpf.kernels.ChangePoints(
                    kernels, locations=[np.random.randint(X.min(), X.max())], steepness=[100]))
            else:
                model = gpf.models.SVGP(likelihood=gpf.likelihoods.StudentT(), inducing_variable=np.linspace(0, max(X), 20), kernel=kernels[0]) 
            return model

        models = []
        for _ in range(n_attempts):
            model = get_model(X, y, model_name, base_kernel_name, custom_kernel)

            data = (X, y)
            loss_fn = model.training_loss_closure(data)

            gpf.utilities.set_trainable(model.q_mu, False)
            gpf.utilities.set_trainable(model.q_sqrt, False)

            variational_vars = [(model.q_mu, model.q_sqrt)]
            natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

            adam_vars = model.trainable_variables
            adam_opt = tf.optimizers.Adam(0.01)

            @tf.function
            def optimisation_step():
                natgrad_opt.minimize(loss_fn, variational_vars)
                adam_opt.minimize(loss_fn, adam_vars)
            
            for i in range(100):
                optimisation_step()
                
            models.append(model)

        # Find model with highest likelihood.
        models.sort(key=lambda m: m.maximum_log_likelihood_objective((X, y)))
        model = models[-1]
        return model

    def call(self, X, y, base_kernel_name="constant", custom_kernel=None):
        """Recursive bisecting function. 

        Arguments:
            X -- index
            y -- observations
            base_kernel_name -- kernel used in the CP kernel

        Returns:
            LOCS, STEEPNESSES -- changepoint LOCationS and their associated STEEPNESSES
        """
        print(X.min(), X.max())
        sample_both = []
        models = {}
        for model_name in ["cp", "gpr"]:
            model = self.get_high_likelihood_model(
                X, y, model_name, base_kernel_name, custom_kernel, 5)

            if self.sampling:
                print("+"*100)
                assign_prior(model, 1, verbose=True)
                model, samples, _, hmc_helper = run_mala(
                    model, step_size=self.stepsize, num_burnin_steps=self.burnin, num_samples=self.samples)

                # Assign median of samples to model
                gpf.utilities.print_summary(model)
                param_values = [np.median(s.numpy(), axis=0) for s in samples]
                for var, map_estimate in zip(hmc_helper.current_state, param_values):
                    if var.shape == (1,):
                        map_estimate = map_estimate.flatten()
                    var.assign(map_estimate)
                gpf.utilities.print_summary(model)
                sample_both.append(samples)
            models[model_name] = model

        cp = models["cp"]
        gpr = models["gpr"]
        location = cp.kernel.locations.numpy()
        steep = cp.kernel.steepness.numpy()
        
        

        LRT = -2 * (gpr.maximum_log_likelihood_objective((X, y)) - cp.maximum_log_likelihood_objective((X, y)))
        # LRT = -2 * (gpr.elbo() -
        #             cp.elbo())
        self.lrts.append(LRT)
        df = len(cp.trainable_parameters) - len(gpr.trainable_parameters)
        p = chi2.sf(LRT, df)
        print(p, LRT, df)

        cp_param = len(cp.trainable_parameters)
        gpr_param = len(gpr.trainable_parameters)

        #bic = cp.log_marginal_likelihood() - gpr.log_marginal_likelihood()
        #bic = cp.elbo() - gpr.elbo() - 0.5*(cp_param - gpr_param)*np.log(len(X))

        print("p", p, "df", df,
              "location", location, "steepness", steep)
        test = [float(location), float(steep), p,
                float(X[0]), float(X[-1]), cp, gpr]
        if self.sampling:
            test += sample_both
        else:
            test += [[], []]
        self.TESTED.append(test)

        # Try splitting
        # The null model is favored and we are done.
        if p > self.pval or np.isnan(p):
            return self.LOCS, self.STEEPNESS
        else:  # Split the signal
            # cutoff when t_0 is out of bounds.
            if min(X) > location or location > max(X):
                return self.LOCS, self.STEEPNESS

            # Check if location not found, else return.
            if int(location) not in list(map(int, self.LOCS)):
                self.LOCS.append(location)
                self.STEEPNESS.append(steep)
            else:
                return self.LOCS, self.STEEPNESS

            # Split: if split out of bounds, ignore.

            try:
                b1 = -5
                b2 = 5
                if location + b1 <= X.min():
                    b1 = -1
                if location + b2 >= X.max():
                    b2 = -1

                split_left = list(map(int, X)).index(int(location+b1))
                split_right = list(map(int, X)).index(int(location+b2))

                X_left, X_right = X[:split_left], X[split_right:]
                y_left, y_right = y[:split_left], y[split_right:]
            except ValueError:
                split = list(map(int, X)).index(int(location))
                X_left, X_right = X[:split], X[split:]
                y_left, y_right = y[:split], y[split:]
            # Recurse if signal is long enough.
            if len(X_left) > 2:
                self.call(X_left, y_left, base_kernel_name)
            if len(X_right) > 2:
                self.call(X_right, y_right, base_kernel_name)
        return self.LOCS, self.STEEPNESS


if __name__ == "__main__":
    X = np.linspace(0, 100, 100).reshape(-1, 1)
    k = gpf.kernels.ChangePoints(kernels=[gpf.kernels.Constant(10), gpf.kernels.Constant(
        10), gpf.kernels.Constant(10)], locations=[40, 70], steepness=[1, 1])
    ys = [(np.random.multivariate_normal(np.zeros(X.shape[0]), k(X)) +
        np.random.normal(0, 0.1, X.shape[0])).reshape(-1, 1) for _ in range(10)]

    segcpgp = SegCPGP(sampling=True)
    locs, steepness = segcpgp.fit(X, ys[0])
