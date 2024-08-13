import os
import numpy as np
from scipy.signal import convolve2d

from sklearn.neural_network import MLPRegressor # for a random function generator
import rasterio as rio # To handle working with geospatial data
import multiprocessing as mp # To parallelize the computation of the NLCD weights

# For creating a smooth random function
from scipy.interpolate import CubicSpline

class SpatialDataset:
    def __init__(
        self,
        dataset_path,
        window_size,
        num_samples,
        len_scale,
        non_linear_confound = False,
        conf_layers = 3,
        conf_hidden = 20,
        seed=42
    ):
        """
        This class is used to generate the dataset for spatial confounding

        Args:
          dataset_path: where the data is stored.
          window_size: the size of the window around the center to consider
          num_samples: the total number of samples to generate for the dataset
          seed: the random seed to use for the dataset
          len_scale: the length scale to use for the weight matrix
        """

        # Load NLCD, NDVI, and unobserved confounder
        with rio.open(os.path.join(dataset_path, "durham_nlcd.tif")) as src:
            nlcd = src.read(1)

        with rio.open(os.path.join(dataset_path, "durham_ndvi.tif")) as src:
            ndvi = src.read(1)

        with rio.open(os.path.join(dataset_path, "durham_synth_unobs_confound.tif")) as src:
            U = src.read(1)

        U = np.flipud(U)
            
        assert nlcd.shape == ndvi.shape
        assert nlcd.shape == U.shape

        # This line converts NLCD to a percentage representation
        # of the land cover class in the window 
        self.nlcd = self._create_one_hot_encoding(nlcd)
        self.ndvi = ndvi
        self.U = U
        self.window_size = window_size

        # Generate a set of num_samples coords that are within the bounds of the image
        # and are not within window_size of the edge of the image
        np.random.seed(seed)
        self.coords = self._sample_coords(nlcd.shape, window_size, num_samples)

        # Reset the random seed
        np.random.seed(seed + 1)
        self.response = self._create_random_function(seed)

        self.wm = self._calc_weight_matrix(window_size, len_scale)

        # Lastly, generate the coefficients
        # A random number is generated for the confounding coefficient
        self.de_coef = -4

        if non_linear_confound:
            # pca = self._fit_nlcd_pca()
            self.confound = RandomMLP(
                                self.nlcd.shape[2],
                                self.nlcd.mean(axis=(0,1)),
                                self.nlcd.std(axis=(0,1)),
                                layers=conf_layers, 
                                hidden_dim=conf_hidden, 
                                random_seed=seed
                            )
        else:
            self.confound = RandomLinear(self.nlcd.shape[2], random_seed=seed)


    def _create_random_function(self, seed):
        """
        This function generates a random function. The weight matrix is multiplied with 
        the neighboring cells, and then that output is passed through this function.

        The result is a non-linear indirect effect.
        """
        x = np.linspace(-1, 1, 10)

        y = np.random.multivariate_normal(0.4-3*x**2, 0.3*np.eye(10), 1)
        # y = np.sort(y)
        y = y.reshape(-1, 1)

        # Now, we arblob:vscode-webview://1ef1mbmj3fe0tko2c05m83hk4q75uf0hdhelglsnnj462nujcjah/3b4462e0-5ceb-4474-a4b3-21edca62c990e going to create a cubic spline as the function
        # This allows the random points to be converted to a smooth function
        cs = CubicSpline(x, y)

        return cs


    def _sample_coords(self, shape, window_size, num_samples):
        """This function samples num_samples coordinates from the image
        and those are the coords used for the dataset."""

        # Set the meshgrid to be the size of the image
        x = np.arange(1 + window_size, shape[0] - window_size)
        y = np.arange(1 + window_size, shape[1] - window_size)
        x, y = np.meshgrid(x, y)
        # Stack the coords so they can be sampled
        coords = np.vstack((x.flatten(), y.flatten())).T

        # Sample num_samples coords from the meshgrid, with
        # no replacement
        samples = np.random.choice(coords.shape[0], num_samples, replace=False)
        coords = coords[samples, :]
        return coords

    def calc_temp(self, ndvi_window, nlcd, U):
        """
        This function uses NDVI, NLCD, and U to return a temperature
        value for the given window.
        
        Arguments:
            ndvi_window: the window of NDVI values
            nlcd_window: the window of NLCD values
            U: the unobserved confounder

        """

        # Get the center of the ndvi_window
        ndvi_center = ndvi_window[self.window_size, self.window_size]

        # Run the ndvi_window through the random function
        indirect_effect = self.response(
            np.sum(self.wm * ndvi_window)
        )[0]

        # Return the temperature value
        return self.de_coef*ndvi_center + indirect_effect + self.confound(nlcd) + U

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        # Get the coordinates
        i, j = self.coords[idx]

        # Return the NLCD
        nlcd = self.nlcd[
            i - self.window_size : i + self.window_size + 1,
            j - self.window_size : j + self.window_size + 1,
            :
        ].mean(axis=(0,1))
        
        # Return the NDVI 
        ndvi = self.ndvi[
            i - self.window_size : i + self.window_size + 1,
            j - self.window_size : j + self.window_size + 1,
        ]

        # Looking at U
        U = self.U[i, j]

        temp = self.calc_temp(ndvi, nlcd, U)
        
        # Create the counterfactuals

        # Counterfactual where only center NDVI is 0
        ndvi_01 = ndvi.copy()
        ndvi_01[self.window_size, self.window_size] = 0
        temp_01 = self.calc_temp(ndvi_01, nlcd, U)

        # Counterfactual where only center NDVI is non-zero
        ndvi_10 = np.zeros_like(ndvi)
        ndvi_10[self.window_size, self.window_size] = ndvi[self.window_size, self.window_size]
        temp_10 = self.calc_temp(ndvi_10, nlcd, U)

        # Counterfactual where NDVI is all 0
        ndvi_00 = np.zeros_like(ndvi)
        temp_00 = self.calc_temp(ndvi_00, nlcd, U)
        
        # Notably, we won't return U, since it is unobserved
        return i/self.ndvi.shape[0], j/self.ndvi.shape[1], nlcd, ndvi, temp, temp_01, temp_10, temp_00

    def _create_one_hot_encoding(self, nlcd):
        # Creating the spatially aggregated one-hot encoding
        cat = len(np.unique(nlcd))
        # Initialize the one-hot encoding vector as zeros.
        one_hot = np.zeros((nlcd.shape[0], nlcd.shape[1], cat))

        # Set the values of the one-hot encoding based on the land use data.
        for i, segment in enumerate(np.unique(nlcd)):
            one_hot[:, :, i] = (nlcd == segment).astype(int)

        return one_hot

    def _task(self, args):
        weight = convolve2d(args[0], args[1], mode="same")
        # weight = (weight - np.mean(weight)) / (np.std(weight) + 1e-8)
        return weight

    def _calc_weight_matrix(self, window_size, length_scale):
        """
        This function calculates the weight matrix for the given window size.
        """
        dist_matrix = np.sqrt(
            np.arange(-window_size, window_size + 1)[np.newaxis, :] ** 2
            + np.arange(-window_size, window_size + 1)[:, np.newaxis] ** 2
        )
        weight_matrix = np.exp(-dist_matrix / length_scale)
        weight_matrix /= weight_matrix.sum()
        return weight_matrix

    def _calc_nlcd_percentage(self, nlcd, window_size):
        """
        This function calculates the percentage of each land cover class
        in the window provided, and returns the percentage of each class
        as an array. 
        """
        one_hot = self._create_one_hot_encoding(nlcd)

        weight_matrix = np.ones((window_size * 2 + 1, window_size * 2 + 1))
        weight_matrix /= weight_matrix.sum()

        # Define the iterable
        iterable = [(one_hot[:, :, i], weight_matrix) for i in range(one_hot.shape[2])]

        nlcd_weight = np.zeros_like(one_hot)
        with mp.Pool(8) as pool:
            for i, result in enumerate(pool.map(self._task, iterable)):
                nlcd_weight[:, :, i] = result

        return nlcd_weight
    
    def calc_causal_effects(self, indices, ndvi_min, ndvi_max, num_bins):
        """
        This function calculates the direct, indirect, and total effects for a subset
        of the dataset based on the given indices.
        """
        # Get the data
        temp_direct, temp_indirect, temp_total = [], [], []
        for idx in indices:
            temp_direct_i, temp_indirect_i, temp_total_i = [], [], []
            i, j = self.coords[idx]
            u = self.U[i, j]
            _, _, nlcd, ndvi, _, _, _, _ = self[idx]
            for ndvi_val in np.linspace(ndvi_min, ndvi_max, num_bins):
                ndvi_window = ndvi.copy()
                # Set the center of the window to the ndvi_val for direct effect
                ndvi_window[self.window_size, self.window_size] = ndvi_val
                temp = self.calc_temp(ndvi_window, nlcd, u)
                temp_direct_i.append(temp)
                # Set the all the pixels to the ndvi_val for total effect
                ndvi_window = np.ones_like(ndvi) * ndvi_val
                temp = self.calc_temp(ndvi_window, nlcd, u)
                temp_total_i.append(temp)
                # Set the center of the window back to the original value for indirect effect
                ndvi_window[self.window_size, self.window_size] = ndvi[self.window_size, self.window_size]
                temp = self.calc_temp(ndvi_window, nlcd, u)
                temp_indirect_i.append(temp)
            temp_direct.append(temp_direct_i)
            temp_indirect.append(temp_indirect_i)
            temp_total.append(temp_total_i)
        
        temp_direct, temp_indirect, temp_total = np.array(temp_direct).squeeze(), \
            np.array(temp_indirect).squeeze(), np.array(temp_total).squeeze()
        direct_effect = np.mean(np.mean(temp_direct, axis=1), axis=0)
        indirect_effect = np.mean(np.mean(temp_indirect, axis=1), axis=0)
        total_effect = np.mean(np.mean(temp_total, axis=1), axis=0)

        return direct_effect, indirect_effect, total_effect


class RandomLinear:
    def __init__(self, dim_in, random_seed=42):
        """
        This class creates a random linear model so that we can create a random function
        to model the confounding effect.
        """
        np.random.seed(random_seed)
        self.coefs = np.random.uniform(0, 1, dim_in)

    def __call__(self, x):
        return np.matmul(x, self.coefs)

class RandomMLP:
    def __init__(self, dim_in, mean, std, layers = 10, hidden_dim = 10, random_seed=42):
        """
        This class creates a random multi-layer perceptron with a sigmoid activation
        function so that we can create a random function to model the confounding effect.
        """
        np.random.seed(random_seed)
        # First layer maps input to hidden layer
        coefs = [np.random.randn(dim_in, hidden_dim)]

        # For each of the layers, we map the hidden layer to itself
        if layers > 1:
            for l in range(layers - 1):
                coefs.append(np.random.randn(hidden_dim, hidden_dim))

        # Fully connected layer last (dim_out = 1)
        coefs.append(np.random.randn(hidden_dim, 1))
        
        self.coefs = coefs
        self.layers = layers
        self.mean = mean
        self.std = std
        self.a = lambda x: 1 / (1 + np.exp(-x)) 

    def __call__(self, x):
        x = (x - self.mean) / self.std
        # This just calls the function on itself
        for l in range(self.layers):
            x = self.a(np.matmul(x, self.coefs[l]))

        # Return final output without activation function
        x = np.matmul(x, self.coefs[-1])
        return x