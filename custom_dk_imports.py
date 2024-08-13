# custom_dk_imports.py
import sys
import os

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union
from torch.utils.data import Dataset

from typing import Tuple
from sklearn.cluster import KMeans

class Erf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.erf(x)

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "softmax": nn.Softmax(),
    "erf": Erf(),
}

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
OPTIMIZERS = {
    'adam': torch.optim.Adam, 
    'sgd': torch.optim.SGD, 
    'rmsprop': torch.optim.RMSprop, 
    'adagrad': torch.optim.Adagrad, 
    'adadelta': torch.optim.Adadelta, 
    'adamw': torch.optim.AdamW,
    'sparseadam': torch.optim.SparseAdam,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'lbfgs': torch.optim.LBFGS,
    'rprop': torch.optim.Rprop,
}

# define an RBF kernel function in numpy
def RBFkernel(x1, x2, l=1.0, sigma_f=1.0, sigma_n=1e-2, noise=False):
    # compute squared distance between points in x1 and x2
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    if noise == True: # If noise=True, add noise to cov matrix by incrementing diagonal elements by (sigma_n)^2, where x1 and x2 points are identical:
        sig_tmp=sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
        for i,x1i in enumerate(x1):
            for j,x2j in enumerate(x2):
                if np.array_equal(x1i,x2j):
                    sig_tmp[i,j]+=sigma_n**2
        return sig_tmp
    else: # if noise=False, the covariance matrix is computed directly (without noise included)
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

# Define necessary parts from nn.py
class DeepKrigingEmbedding2D(nn.Module):
    def __init__(self, K: int):
        super(DeepKrigingEmbedding2D, self).__init__()
        self.K = K
        self.num_basis = [(9*2**(h-1)+1)**2 for h in range(1,self.K+1)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        knots_1d = [torch.linspace(0, 1, int(np.sqrt(i))).to(self.device) for i in self.num_basis]
        N = s.shape[0]
        phi = torch.zeros(N, sum(self.num_basis)).to(self.device)
        K = 0
        for res, num_basis_res in enumerate(self.num_basis):
            theta = 1 / np.sqrt(num_basis_res) * 2.5
            knots_s1, knots_s2 = torch.meshgrid(knots_1d[res], knots_1d[res], indexing='ij')
            knots = torch.stack((knots_s1.flatten(), knots_s2.flatten()), dim=1).to(self.device)
            d = torch.cdist(s, knots) / theta
            mask = (d >= 0) & (d <= 1)
            weights = torch.zeros_like(d)
            weights[mask] = ((1 - d[mask]) ** 6 * (35 * d[mask] ** 2 + 18 * d[mask] + 3) / 3)
            phi[:, K:K + num_basis_res] = weights
            K += num_basis_res
        return phi

class MLP(nn.Module):
    def __init__(self, input_dim: int, num_hidden_layers: int, hidden_dims: Union[int, List[int]], 
                 batch_norm: bool = False, p_dropout: float = 0.0, activation: str = "relu") -> None:
        super(MLP, self).__init__()
        self.input_dim: int = input_dim
        self.num_hidden_layers: int = num_hidden_layers
        self.hidden_dims: Union[int, List[int]] = hidden_dims
        self.batch_norm: bool = batch_norm
        self.p_dropout: float = p_dropout
        self.activation: str = activation
        self._validate_inputs()
        self._build_layers()
    
    def _validate_inputs(self) -> None:
        if isinstance(self.hidden_dims, List):
            assert len(self.hidden_dims) == self.num_hidden_layers, \
                "Number of hidden layers must match the length of hidden dimensions"
        assert self.p_dropout >= 0 and self.p_dropout < 1, "Dropout probability must be in [0, 1)"
        assert self.activation in ACTIVATIONS, "Activation function must be one of {}".format(ACTIVATIONS.keys())
    
    def _build_layers(self) -> None:
        layers = []
        if isinstance(self.hidden_dims, int):
            hidden_dims = [self.hidden_dims] * self.num_hidden_layers
        else:
            hidden_dims = self.hidden_dims
        for i in range(self.num_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(ACTIVATIONS[self.activation])
            if self.p_dropout > 0:
                layers.append(nn.Dropout(self.p_dropout))
        if self.num_hidden_layers > 0:
            last_layer_dim = hidden_dims[-1] if isinstance(hidden_dims, List) else hidden_dims
        else:
            last_layer_dim = self.input_dim
        layers.append(nn.Linear(last_layer_dim, 1))
        self.mlp_layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(x)

class DeepKrigingMLP(MLP):
    def __init__(self, input_dim: int, num_hidden_layers: int, hidden_dims: Union[int, List[int]], K: int = 1,
                 batch_norm: bool = False, p_dropout: float = 0.0, activation: str = "relu") -> None:
        self.K: int = K
        super(DeepKrigingMLP, self).__init__(input_dim, num_hidden_layers, hidden_dims, batch_norm, 
                                             p_dropout, activation)
        self._build_layers()
    
    def _build_layers(self) -> None:
        super(DeepKrigingMLP, self)._build_layers()
        self.embedding = DeepKrigingEmbedding2D(self.K)
        out_feature = self.hidden_dims[0] if isinstance(self.hidden_dims, List) else self.hidden_dims
        self.mlp_layers[0] = nn.Linear(
            self.mlp_layers[0].in_features + sum(self.embedding.num_basis), out_feature
        )
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        phi = self.embedding(s)
        return self.mlp_layers(torch.cat([phi], dim=1))

class MSELoss(torch.nn.modules.loss._Loss):
    """
    Mean Squared Error Loss
    """
    def __init__(self) -> None:
        super(MSELoss, self).__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true) ** 2)

# Define necessary parts from trainers.py
class NewLoss(nn.modules.loss._Loss):
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray) -> None:
        super(NewLoss, self).__init__()
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

        self.observed = np.arange(self.X_train.shape[0])
        X_combined = np.vstack((X_train, X_test))
        K_combined = RBFkernel(X_combined, X_combined, l=1., sigma_f=1., noise=True, sigma_n=0.01)
        self.K_inv_pt = torch.inverse(torch.tensor(K_combined, dtype=torch.float32))

        M = np.zeros(X_train.shape[0] + X_test.shape[0])
        M[self.observed] = 1
        self.M = torch.tensor(M, dtype=torch.float32)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        observation_matching = MSELoss()

        # Compute loss over mini-batch instead of the entire dataset
        loss_1 = 0.5 * torch.matmul(torch.matmul(y_pred.T, self.K_inv_pt[:y_pred.shape[0], :y_pred.shape[0]]), y_pred) / y_pred.shape[0]

        loss_2 = observation_matching(self.y_train[:y_pred.shape[0]], y_pred[:self.observed.shape[0]])
        loss_3 = -torch.mean(self.M[:y_pred.shape[0]] * torch.log(torch.sigmoid(y_pred)) + (1 - self.M[:y_pred.shape[0]]) * torch.log(1 - torch.sigmoid(y_pred)))

        loss = loss_1 + loss_2 + loss_3
        return loss
    

class SpatialDataset(torch.utils.data.Dataset):
    def __init__(self, s: np.ndarray, y: np.ndarray) -> None:
        self.s: np.ndarray = s
        self.y: np.ndarray = y

    def _validate_and_preprocess_inputs(self) -> None:
        assert len(self.s) == len(self.y), \
            "Spatial features and targets must be the same length"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx) -> Tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        batch = torch.from_numpy(self.s[idx]).float() if isinstance(self.s[idx], np.ndarray) else torch.tensor(self.s[idx]), \
            torch.from_numpy(self.y[idx]).float() if isinstance(self.y[idx], np.ndarray) else torch.tensor(self.y[idx])
        return batch
    
def train_val_test_split(t: List, x: np.ndarray, s: np.ndarray, y: np.ndarray, train_size: float = 0.7,
                         val_size: float = 0.15, test_size: float = 0.15, shuffle: bool = True, random_state: int = 2020, 
                         block_sampling: bool = False, num_blocks: int = 50, return_test_indices: bool = False, 
                         graph_input: bool = False) -> Tuple:
    """
    Splits the dataset into training, validation, and test sets
    If shuffle is set to True, the data will be shuffled before splitting
    If block_sampling is set to True, the data will first be clustered into the specified number of blocks using K-means
    and then splitted into training, validation, and test sets by sampling from the blocks
    """
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1"
    assert all([len(feat) == len(y) for feat in t]), "Interventions and targets must be the same length"
    assert len(x) == len(y), "Confounder and targets must be the same length"
    assert len(s) == len(y), "Spatial features and targets must be the same length"

    if shuffle:
        np.random.seed(random_state)
        idx = np.random.permutation(len(y))
        t = [feat[idx] for feat in t]
        x, s, y = x[idx], s[idx], y[idx]
    
    if block_sampling:
        kmeans = KMeans(n_clusters=num_blocks, random_state=random_state).fit(s)
        block_labels = kmeans.labels_
        block_indices = np.arange(num_blocks)
        train_blocks = np.random.choice(block_indices, size=int(train_size * num_blocks), replace=False)
        block_indices = np.setdiff1d(block_indices, train_blocks)
        val_blocks = np.random.choice(block_indices, size=int(val_size * num_blocks), replace=False)
        test_blocks = np.setdiff1d(block_indices, val_blocks)
        train_idx = np.where(np.isin(block_labels, train_blocks))[0]
        val_idx = np.where(np.isin(block_labels, val_blocks))[0]
        test_idx = np.where(np.isin(block_labels, test_blocks))[0]

        t_train, x_train, s_train = [feat[train_idx] for feat in t], x[train_idx], s[train_idx]
        t_val, x_val, s_val = [feat[val_idx] for feat in t], x[val_idx], s[val_idx]
        t_test, x_test, s_test = [feat[test_idx] for feat in t], x[test_idx], s[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    else:
        train_idx = int(train_size * len(y))
        val_idx = int((train_size + val_size) * len(y))

        t_train, x_train, s_train = [feat[:train_idx] for feat in t], x[:train_idx], s[:train_idx]
        t_val, x_val, s_val = [feat[train_idx:val_idx] for feat in t], x[train_idx:val_idx], s[train_idx:val_idx]
        t_test, x_test, s_test = [feat[val_idx:] for feat in t], x[val_idx:], s[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]

    if not graph_input:
        res = tuple([
            SpatialDataset(t_train, x_train, s_train, y_train), 
            SpatialDataset(t_val, x_val, s_val, y_val),
            SpatialDataset(t_test, x_test, s_test, y_test)
        ])
    else:
        res = tuple([
            GraphSpatialDataset(t_train, x_train, s_train, y_train), 
            GraphSpatialDataset(t_val, x_val, s_val, y_val),
            GraphSpatialDataset(t_test, x_test, s_test, y_test)
        ])
    if return_test_indices:
        res += (test_idx,) if block_sampling else (idx[val_idx:],)
    return res

def simple_train_val_test_split(s: np.ndarray, y: np.ndarray, train_size: float = 0.7,
                                val_size: float = 0.15, test_size: float = 0.15, 
                                shuffle: bool = True, random_state: int = 2020) -> Tuple:
    """
    Splits the dataset into training, validation, and test sets based on spatial coordinates.
    """
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1"
    assert len(s) == len(y), "Spatial features and targets must be the same length"

    if shuffle:
        np.random.seed(random_state)
        idx = np.random.permutation(len(y))
        s, y = s[idx], y[idx]
    
    train_idx = int(train_size * len(y))
    val_idx = int((train_size + val_size) * len(y))

    s_train, s_val, s_test = s[:train_idx], s[train_idx:val_idx], s[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]

    train_dataset = SpatialDataset(s_train, y_train)
    val_dataset = SpatialDataset(s_val, y_val)
    test_dataset = SpatialDataset(s_test, y_test)

    return train_dataset, val_dataset, test_dataset

import logging
class Trainer:
    def __init__(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
                 optim: str, optim_params: dict, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 model_save_dir: str = None, model_name: str = 'model.pt', 
                 loss_fn: nn.modules.loss._Loss = nn.MSELoss(), device: torch.device = torch.device('cpu'), 
                 epochs: int = 100, patience: int = 10, logger: logging.Logger = logging.getLogger("Trainer"), 
                 wandb: object = None) -> None:
        self.model: nn.Module = model
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.val_loader: torch.utils.data.DataLoader = val_loader
        self.optim: str = optim
        self.optim_params: dict = optim_params
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = lr_scheduler
        self.model_save_dir: str = model_save_dir
        self.model_name: str = model_name
        self.loss_fn: nn.modules.loss._Loss = loss_fn
        self.device: torch.device = device
        self.epochs: int = epochs
        self.patience: int = patience
        self.logger: logging.Logger = logger
        self.wandb: object = wandb
        
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        if self.model_save_dir is not None and not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self._register_wandb_params()
        self._set_optimizer()

    def _register_wandb_params(self) -> None:
        if self.wandb is not None:
            self.wandb.config.update({
                "optimizer": self.optim,
                "optimizer_params": self.optim_params,
                "loss_fn": self.loss_fn,
                "epochs": self.epochs,
                "patience": self.patience
            }, allow_val_change=True)
    
    def _set_optimizer(self) -> None:
        OPTIMIZERS = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            # Add other optimizers as needed
        }
        if self.optim not in OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {self.optim}")
        self.optimizer = OPTIMIZERS[self.optim](self.model.parameters(), **self.optim_params)

    def train(self) -> None:
        best_loss = float('inf')
        best_model_state_dict = None
        trigger_times = 0
        
        self.logger.info("Training started:\n")
        for epoch in range(self.epochs):
            train_loss = self._train_step()
            val_loss = self.validate()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if val_loss > best_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    self.logger.info("Early stopping - patience reached")
                    if best_model_state_dict is not None:
                        self.logger.info("Restoring the best model")
                        self.model.load_state_dict(best_model_state_dict)
                    if self.model_save_dir is not None:
                        self.logger.info("Saving the best model")
                        torch.save(best_model_state_dict, os.path.join(self.model_save_dir, self.model_name))
                    break
            else:
                trigger_times = 0
                best_loss = val_loss
                best_model_state_dict = self.model.state_dict()

    def validate(self) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, y in self.val_loader:
                s, y = s.to(self.device), y.to(self.device).view(-1)
                y_pred = self.model(s).float()
                val_loss += self.loss_fn(y_pred, y.float()).item()
        return val_loss / len(self.val_loader)

    def _train_step(self) -> float:
        self.model.train()
        train_loss = 0.0
        for s, y in self.train_loader:
            s, y = s.to(self.device), y.to(self.device).view(-1)
            self.optimizer.zero_grad()
            y_pred = self.model(s).float()
            loss = self.loss_fn(y_pred, y.float())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)
