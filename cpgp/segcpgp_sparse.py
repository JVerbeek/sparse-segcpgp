import sys
import os
import gpflow as gpf
import numpy as np
np.random.seed(314)
f64 = gpf.utilities.to_default_float
sys.path.append(os.path.expanduser("~")+"/changepoint-gp/")
from cpgp.hmc import assign_prior, run_mala
from cpgp.spectral_mixture import SpectralMixture
from scipy.stats.distributions import chi2
import tensorflow as tf
from scipy.cluster.vq import kmeans

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

    def __init__(self, n_ind=10, pval=0.1, df=1, stepsize=0.1, burnin=1000, samples=5000, logging=False, sampling=False) -> None:
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
        self.n_ind = n_ind

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

    def get_high_likelihood_model(self, X, y, model_name, base_kernel_name, custom_kernel, n_attempts=3, fit_noise=True, inducing_points=np.array([])):
        """The LML has no guarantees that it does not end up in a local optimum.
        We select the model with the highest likelihood after a number of attempts. 

        Arguments:
            attempts -- number of attempts to make

        Returns: 
            model -- model with the highest likelihood.
        """
        def get_model(X, y, model_name, base_kernel_name, custom_kernel):
            """Utility function for getting either a GPR or a CP based on the model_name string."""
            n_inducing = self.n_ind
            inducing_var = np.concatenate((np.random.choice(X.flatten(), n_inducing - len(inducing_points)), np.array(inducing_points).flatten())).reshape(-1, 1)
            #inducing_var = np.random.choice(X.flatten(), n_inducing).reshape(-1, 1) # if not inducing_points else inducing_points
            if not custom_kernel:
                kernels = [get_kernel(base_kernel_name, X, y), get_kernel(base_kernel_name, X, y)]
            else:
                kernels = [custom_kernel, custom_kernel]
            if model_name == "cp":
                model = gpf.models.SGPR((X, y), inducing_variable=inducing_var, kernel=gpf.kernels.ChangePoints(
                    kernels, locations=[np.random.randint(X.min(), X.max())], steepness=[1]))
            else:
                model = gpf.models.SGPR((X, y), kernel=kernels[0], inducing_variable=inducing_var)
            return model

        models = []
        for _ in range(n_attempts):
            model = get_model(X, y, model_name, base_kernel_name, custom_kernel)

            # We want to manually fit the noise sometimes.
            if "noise" in base_kernel_name and fit_noise:
                gpf.set_trainable(model.likelihood.variance, False)
                model.likelihood.variance.assign(0.00001)
            gpf.set_trainable(model.inducing_variable, False)
            optimizer = gpf.optimizers.Scipy()
            optimizer.minimize(model.training_loss, model.trainable_variables)
            models.append(model)

        # Find model with highest likelihood.
        models.sort(key=lambda m: m.elbo())
        model = models[-1]
        return model

    def call(self, X, y, base_kernel_name="constant", custom_kernel=None, inducing_points=[]):
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
                X, y, model_name, base_kernel_name, custom_kernel, 5, inducing_points=inducing_points)
            models[model_name] = model

        cp = models["cp"]
        gpr = models["gpr"]
        location = cp.kernel.locations.numpy()
        steep = cp.kernel.steepness.numpy()
        
        
        # Model selection time
        LRT = -2 * (gpr.elbo() -
                    cp.elbo())
        self.lrts.append(LRT)
        df = len(cp.trainable_parameters) - len(gpr.trainable_parameters)
        p = chi2.sf(LRT, df)
        
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
            

            inducing_points = cp.inducing_variable.Z.numpy()
            inducing_left = inducing_points[inducing_points <= split_left].reshape(-1, 1)
            inducing_right = inducing_points[inducing_points >= split_right].reshape(-1, 1)
                
            # Recurse
            if len(X_left) > 2:
                self.call(X_left, y_left, base_kernel_name, inducing_points=inducing_left)
            if len(X_right) > 2:
                self.call(X_right, y_right, base_kernel_name, inducing_points=inducing_right)
        return self.LOCS, self.STEEPNESS


if __name__ == "__main__":
    X = np.linspace(0, 100, 100).reshape(-1, 1)
    k = gpf.kernels.ChangePoints(kernels=[gpf.kernels.Constant(10), gpf.kernels.Constant(
        10), gpf.kernels.Constant(10)], locations=[40, 70], steepness=[1, 1])
    ys = [(np.random.multivariate_normal(np.zeros(X.shape[0]), k(X)) +
        np.random.normal(0, 0.1, X.shape[0])).reshape(-1, 1) for _ in range(10)]

    segcpgp = SegCPGP(sampling=True)
    locs, steepness = segcpgp.fit(X, ys[0])
