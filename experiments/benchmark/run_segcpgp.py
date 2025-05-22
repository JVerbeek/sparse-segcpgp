import numpy as np
import argparse
import yaml
import json
import sys
import time
sys.path.append("/home/janneke/changepoint-gp/")
from cpgp.segcpgp_sparse import SegCPGP as SparseSegCPGP

parser = argparse.ArgumentParser(description="Dataset path.")
parser.add_argument(
    '-d', '--dataset',
    required=True,
    help='Path to the dataset'
)
parser.add_argument(
    '-p', '--parameters',
    required=True,
    help='Parameters.'
)

parser.add_argument(
    '-r', '--results',
    default="test",
    required=True,
    help='Results path.'
)

args = parser.parse_args()
print(args.results)

# Load parameters
with open(args.parameters, 'r') as file:
    parameters = yaml.safe_load(file)

# Load data   
data = np.load(args.dataset)
X = data["X"].reshape(-1, 1)
y = data["y"].reshape(-1, 1)

# Fit model
segcpgp = SparseSegCPGP(n_ind=parameters["n_inducing"], pval=parameters["pval"])
t = time.time()
segcpgp.fit(X, y, base_kernel_name=parameters["kernel"])

dt = time.time() - t
locations = segcpgp.LOCS

# Write results and metadata to JSON
results = {}
results["runtime"] = dt
results["dataset"] = args.dataset.split("/")[-1].split(".")[0]
results["locations"] = [l.tolist() for l in locations]
results.update(parameters)

with open(f'results/{args.results}_{results["dataset"]}.json', 'a') as fp:
    json.dump(results, fp)
    fp.write("\n")