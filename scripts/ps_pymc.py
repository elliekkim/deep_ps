"""
This script fits the preferential sampling model using PyMC, then saves the results
for future use.
"""
import os
import argparse
import numpy as np
import pymc as pm


def main(args):

    # Load the data
    X_obs, y_obs, obs, X_all = load_data(args.file_path, args.test)

    # Define the model.
    with pm.Model() as ps_model:

        # Specify the covariance function.
        nu = pm.Gamma('nu', alpha = 3, beta = 1)
        ls = pm.Gamma("ls", 10, 1.5)
        cov_func = nu**2 * pm.gp.cov.Exponential(2, ls=ls)

        # Specify the GP.  The default mean function is `Zero`.
        gp = pm.gp.Marginal(cov_func=cov_func)

        # Place a GP prior over the function f.
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        y_ = gp.marginal_likelihood("y", X=X_obs, y=y_obs, sigma=sigma)

        fcond = gp.conditional("fcond", Xnew=X_all, pred_noise=True)
        theta0 = pm.Normal("theta0", mu=-1, sigma=1)
        theta1 = pm.Normal("theta1", mu=-1, sigma=1)
        intensity = pm.math.exp(theta0 + theta1 * fcond)
        lam = pm.Poisson("lam", mu=intensity, observed = obs)

        # this line calls an optimizer to find the MAP
        mp = pm.find_MAP(include_transformed=True)
        
    print("Model fit. Saving results, now.")

    with ps_model:
        mu, var = gp.predict(X_all, point=mp)

    # Save the predictions
    np.save(os.path.join(args.target_path, 'ps_mu.npy'), mu)
    np.save(os.path.join(args.target_path, 'ps_var.npy'), var)

    # Save the mp results, too.
    np.save(os.path.join(args.target_path, 'ps_mp.npy'), mp)


def load_data(file_path, test):

    # Load data and extract coordinates and temperature
    data = np.load(file_path)
    lat, lon = np.indices(data.shape)
    coords = np.column_stack([lat.ravel(), lon.ravel()])

    y = data.flatten()
    y_mu = np.nanmean(y)
    y_std = np.nanstd(y)
    y = (y - y_mu) / y_std

    obs = ~np.isnan(y)

    coords_obs = coords[obs]
    y_obs = y[obs]

    return coords_obs, y_obs, obs, coords

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Fit the preferential sampling model using PyMC.')

    # Add a test flag
    parser.add_argument('--test', action='store_true', help='Run the script in test mode.')

    # Add a file_path
    parser.add_argument('--file_path', type=str, help='Path to the data file.')

    # Add a target path in which to save the results.
    parser.add_argument('--target_path', type=str, help='Path to save the results.')

    # Parse the arguments
    args = parser.parse_args()
    
    main(args)