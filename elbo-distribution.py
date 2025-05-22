import gpflow as gpf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from tqdm import tqdm 
import pandas as pd

X = np.linspace(0, 100, 100).reshape(-1, 1)
y = np.random.normal(0, 0.1, 100).reshape(-1, 1)

plt.plot(X, y)
plt.show()

res = {"nocp": [], "cp": []}

for i in tqdm(range(500)):
    for model in ["nocp", "cp"]:
        if model == "cp":
            k2 = gpf.kernels.ChangePoints([gpf.kernels.RBF() for i in range(2)], locations=[50], steepness=100)
            model1 = gpf.models.SGPR((X, y), kernel=k2, inducing_variable=np.random.choice(X.flatten(), 20).reshape(-1, 1))
        else:
            model1 = gpf.models.SGPR((X, y), kernel=gpf.kernels.RBF(), inducing_variable=np.random.choice(X.flatten(), 20).reshape(-1, 1))
        opt = gpf.optimizers.Scipy()
        opt.minimize(model1.training_loss, model1.trainable_variables)

        res[model].append(model1.elbo().numpy())
        
res_arr = pd.DataFrame(res).to_numpy()

lrt = -2 * (res_arr[:,0] - res_arr[:,1])
plt.hist(lrt, bins=50)
plt.show()
    
    

