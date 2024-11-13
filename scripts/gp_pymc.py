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
    X, M, y = load_data(args.file_path, args.test)

    # Define the model.
    with pm.Model() as ps_model:

        # Specify the covariance function.
        nu = pm.HalfCauchy('nu', beta=2)
        ls = pm.Gamma("ls", 10, 1.5) #mu=20, sigma=10)
        cov_func = nu**2 * pm.gp.cov.Exponential(2, ls=ls)

        # Specify the GP.  The default mean function is `Zero`.
        gp = pm.gp.Latent(cov_func=cov_func)

        # Place a GP prior over the function f.
        sigma = pm.HalfNormal("sigma", sigma=1)

        phi = gp.prior("phi", X=X)
        y_ = pm.Normal("y", mu=phi, sigma=sigma, observed=y)

        # this line calls an optimizer to find the MAP
        trace = pm.sample(1000, nuts_sampler='nutpie', tune=1000, chains=1, random_seed=42, return_inferencedata=True)

    # Save the results for future analysis.
    trace.to_netcdf(os.path.join(args.target_path, 'gp_results.nc'))


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

    if test:
        # If we are testing, we only want to select a subsample to train on.
        obs_coords = coords[obs]
        obs_y = y[obs]

        unobs_coords = coords[~obs]
        unobs_y = y[~obs]

        # Sample 100 unobserved coords
        unobs_vals = np.random.choice(len(unobs_coords), 20, replace=False)
        sampled_coords = unobs_coords[unobs_vals]
        sampled_y = unobs_y[unobs_vals]
        X = np.vstack([obs_coords, sampled_coords])
        y = np.hstack([obs_y, sampled_y])

        y = np.ma.masked_array(y, mask=np.isnan(y))
        obs = ~y.mask

        return X, obs, y

    else:
        y = np.ma.masked_array(y, mask=~obs)

        return coords, obs, y

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