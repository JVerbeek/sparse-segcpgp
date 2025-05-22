import argparse
import os
import yaml
import numpy as np
import gpflow as gpf
from tqdm import tqdm


def valid_directory(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory")
    
    
def mean_change(ndp, noise_var=0.01, locs=[20, 60], steepnesses=[1, 1]):
    """Generate dataset with mean changes and Gaussian noise. Data is generated from a changepoint Constant kernel, with changepoints at locations specified in locs
    and steepnesses as specified in steepnesses. 

    Arguments:
        ndp -- number of data points generated

    Keyword Arguments:
        noise_var -- variance of the noise (default: {0.01})
        locs -- locations of the changepoints (default: {[20, 60]})
        steepnesses -- steepnesses of the changepoints (default: {[1, 1]})

    Returns:
        X, y, cpk, noise_var -- X is an index and y are the associated observations. cpk is the changepoint kernel used to generate the dataset. 
    """
    kernels = [gpf.kernels.Constant(1) for c in range(len(locs) + 1)]
    cpk = gpf.kernels.ChangePoints(kernels, locations=locs, steepness=steepnesses)
    X = np.linspace(0, ndp, ndp).reshape(-1, 1)
    y = np.random.multivariate_normal(mean=np.zeros(X.shape[0]), cov=cpk(X)).reshape(-1, 1)
    y += np.random.normal(0, noise_var, size=y.shape)
    return X, y

def trend_change(ndp, noise_var=0.5, locs=[20, 60]):
    """Generate dataset with trend changes and Gaussian noise. This is done by making a piecewise linear function, where the sign of the slope changes at each changepoint.
    The changes are at the locations specified by the locs kwarg. 

    Arguments:
        ndp -- number of data points generated

    Keyword Arguments:
        noise_var -- variance of the noise, should be substantial enough to accommodate for the scale of the data. (default: {0.5})
        locs -- locations at which changes are located. (default: {[20, 60]})

    Returns:
        X, y, None, noise var -- X is an index and y are the associated observations. 
    """
    X = np.linspace(0, ndp, ndp).reshape(-1, 1)
    y = np.zeros(X.shape)
    last = 0
    for i in range(len(locs)-1):
        rand = np.random.uniform(-2, 0) if i % 2 == 0 else np.random.uniform(0, 2)  # Let's switch signs every time
        l1, l2 = locs[i], locs[i + 1]
        y[l1:l2] = ((rand * np.arange(len(y[l1:l2]))) + last).reshape(-1, 1)
        last = y[l2-1]
        
    y += np.random.normal(0, noise_var, size=y.shape)
    return X, y

def var_change(ndp, locs=[20, 60], steepnesses=[1, 1]):
    """Generate dataset with variance changes and optional Gaussian noise. Datasets are sampled from a changepoint-Matern12 kernel,
    with changes at locations specified by locs, and steepnesses specified by steepnesses. 

    Arguments:
        ndp -- number of datapoints in the dataset

    Keyword Arguments:
        noise_var -- variance of the noise (default: {0.01})
        locs -- locations of the changepoints (default: {[20, 60]})
        steepnesses -- steepnesses of the changepoints (default: {[1, 1]})

    Returns:
        X, y, cpk, noise_var -- X is an index and y are the associated observations. cpk is the changepoint kernel used to generate the dataset. 
    """
    variances = [1 if i % 2 == 0 else np.random.randint(3, 20) for i in range(len(locs) + 1)]
    kernels = [gpf.kernels.Matern12(variances[i]) for i in range(len(locs) + 1)]
    cpk = gpf.kernels.ChangePoints(kernels, locations=locs, steepness=steepnesses)
    X = np.linspace(0, ndp, ndp).reshape(-1, 1)
    y = np.random.multivariate_normal(mean=np.zeros(X.shape[0]), cov=cpk(X)).reshape(-1, 1)
    return X, y

def per_change(ndp, noise_var=0.01, locs=[20, 60]):
    """Generate dataset with periodicity changes by creating a sine wave with a randomly sampled angular velocity. Gaussian noise is optional.
    Periodicity changes happen at locations given in the locs argument.

    Arguments:
        ndp -- number of data points to be generated.

    Keyword Arguments:
        noise_var -- variance of the noise (default: {0.01})
        locs -- locations of the changepoints (default: {[20, 60]})

    Returns:
        X, y, None, noise_var -- X is an index and y are the associated observations.
    """
    X = np.linspace(0, ndp, ndp).reshape(-1, 1) 
    y = np.zeros(X.shape)

    for i in range(len(locs) - 1):
        rand = np.random.randint(1, 100)
        l1, l2 = locs[i], locs[i + 1]
        y[l1:l2] = np.sin(rand * X[l1:l2])
    y += np.random.normal(0, noise_var, size=y.shape)
    return X, y


parser = argparse.ArgumentParser(description="Script that requires a directory.")
parser.add_argument(
    '-d', '--directory',
    type=valid_directory,
    required=True,
    help='Path to the data directory'
)

parser.add_argument(
    '-p', '--parameters',
    required=True,
    help='Number of datapoints to generate'
)

args = parser.parse_args()
dir = args.directory

with open(args.parameters, 'r') as file:
    parameters = yaml.safe_load(file)
    
noise_var = parameters["noise_var"]
locations = parameters["locations"]
steepnesses = parameters["steepness"]
num_datapoints = parameters["num_datapoints"]
num_datasets = parameters["num_datasets"]


datasets = tqdm(["mean", "trend", "per", "var"])
for name in datasets:
    datasets.set_description(f"Making {name} datasets...")
    for i in tqdm(range(num_datasets)):
        if name == "mean":
            X, y = mean_change(num_datapoints, noise_var, locs=locations, steepnesses=steepnesses)
        if name == "var":
            X, y = var_change(num_datapoints, locs=locations, steepnesses=steepnesses)
        if name == "per":
            X, y = per_change(num_datapoints, noise_var, locs=locations)
        else:
            X, y = trend_change(num_datapoints, noise_var, locs=locations)
            
        np.savez(f"{dir}/{name}/{name}-{i}", X=X, y=y)