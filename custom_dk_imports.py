# custom_dk_imports.py
import sys
import os

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union
from torch.utils.data import Dataset
import time

from typing import Tuple
from sklearn.cluster import KMeans

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import scipy
from scipy.stats import norm

from torch.optim.lr_scheduler import LRScheduler, StepLR


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
    "identity": nn.Identity(),
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

# define an RBF kernel function in pytorch
def RBFkernel(x1, x2, l=1.0, sigma_f=1.0, sigma_n=1e-2, noise=False):
    # Compute squared distance between points in x1 and x2 using PyTorch
    sqdist = torch.sum(x1**2, dim=1).reshape(-1, 1) + torch.sum(x2**2, dim=1) - 2 * torch.mm(x1, x2.T)
    
    # Compute covariance matrix
    K = sigma_f**2 * torch.exp(-0.5 / l**2 * sqdist)
    
    if noise:
        # Add noise to the diagonal elements where x1 and x2 are identical
        diag_indices = torch.arange(x1.shape[0])
        K[diag_indices, diag_indices] += sigma_n**2  # Add noise to diagonal elements
    
    return K

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
        self.input_dim: int = input_dim
        self._build_layers()
    
    def _build_layers(self) -> None:
        super(DeepKrigingMLP, self)._build_layers()
        # self.embedding = DeepKrigingEmbedding2D(self.K)
        out_feature = self.hidden_dims[0] if isinstance(self.hidden_dims, List) else self.hidden_dims
        self.mlp_layers[0] = nn.Linear(
            self.input_dim, out_feature
        )
    
    # def forward(self, s: torch.Tensor) -> torch.Tensor:
    #     phi = self.embedding(s)
    #     return self.mlp_layers(torch.cat([phi], dim=1))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(phi)

class GeneralizedPropensityScoreModel(MLP):
    """
    Generalized Propensity Score Model

    Arguments
    --------------
    input_dim: int, The dimension of the covariate X + spatial information s
    """
    def __init__(self, input_dim: int, num_hidden_layers: int, hidden_dims: Union[int, List[int]], 
                 batch_norm: bool = False, p_dropout: float = 0.0, activation: str = "relu") -> None:
        super(GeneralizedPropensityScoreModel, self).__init__(input_dim, num_hidden_layers, 
            hidden_dims,batch_norm, p_dropout, activation)
        if num_hidden_layers > 0:
            last_hidden_dim = hidden_dims[-1] if isinstance(hidden_dims, List) else hidden_dims
        else:
            last_hidden_dim = input_dim
        self.mlp_layers[-1] = nn.Linear(last_hidden_dim, 2)
    
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        output = self.mlp_layers(torch.cat([x, s], dim=1))
        mean = output[:, 0]
        var = torch.exp(output[:, 1])
        return mean, var
    
    def generate_propensity_score(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, var = self.forward(x, s)
            return norm.pdf(t.cpu().numpy(), mean.cpu().numpy(), torch.sqrt(var).cpu().numpy())

class MSELoss(torch.nn.modules.loss._Loss):
    """
    Mean Squared Error Loss
    """
    def __init__(self) -> None:
        super(MSELoss, self).__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true) ** 2)
    
class NewLoss(nn.modules.loss._Loss):
    def __init__(self, s_all: np.ndarray, observed_indices: np.ndarray, y_all: np.ndarray) -> None:
        super(NewLoss, self).__init__()
        self.s_all = torch.tensor(s_all, dtype=torch.float32).clone().detach().requires_grad_(True)
        self.y_all = torch.tensor(y_all, dtype=torch.float32).clone().detach().requires_grad_(True)

        # Default mask for the entire grid (in case no specific mask is passed)
        M = np.zeros(len(s_all))
        M[observed_indices] = 1
        self.M = torch.tensor(M, dtype=torch.float32)

        self.alpha = nn.Parameter(torch.tensor(-1.0, requires_grad=True))  # Initialize alpha
        self.beta = nn.Parameter(torch.tensor(-1.0, requires_grad=True))   # Initialize beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, M: torch.Tensor = None) -> torch.Tensor:
        observation_matching = nn.MSELoss()

        # Use the provided mask `M` if given; otherwise, default to `self.M`
        M_batch = M if M is not None else self.M
        
        # Compute `sig` based on `y_pred` (batch or full grid)
        sig = torch.sigmoid(self.alpha + self.beta * y_pred)
        # print(f"Gradients of alpha and beta: {self.alpha.grad}, {self.beta.grad}")

        # Ensure M_batch and sig are the same size
        assert M_batch.shape[0] == sig.shape[0], f"Size mismatch: M_batch ({M_batch.shape[0]}) and sig ({sig.shape[0]})"

        # MSE over all points
        loss_observation = observation_matching(y_pred, y_true)

        # BCE over all points
        loss_selection_bias = -torch.mean(
            M_batch * torch.log(sig) + (1 - M_batch) * torch.log(1 - sig)
        )

        # Weighted loss function with scaling terms
        # loss = loss_observation + self.lambda_mse * loss_spatial_matching + self.lambda_bce * loss_selection_bias
        loss = loss_observation + loss_selection_bias
        return loss

import logging
class BaseTrainer(ABC):
    """
    Template for trainers

    Arguments
    --------------
    model: torch.nn.Module, the model to train
    data_generators: Dict, a dict of dataloaders where keys must be 'train', 'val', and 'test'
    optim: str, the name of the optimizer for training
    optim_params: Dict, a dict of parameters for the optimizer
    lr_scheduler: torch.optim.lr_scheduler, the learning rate scheduler for training
    model_save_dir: str, the directory to save the trained model. If None, the model will not be saved
    model_name: str, the name of the trained model
    loss_fn: torch.nn.modules.loss._Loss, the loss function for optimizing the model
    device: torch.device, the device to train the model on
    epochs: int, the number of epochs to train the model for
    patience: int, the number of epochs to wait before early stopping
    logger: logging.Logger, an instance of logging.Logger for logging messages, errors, exceptions
    """
    @abstractmethod
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None, model_save_dir: str = None, model_name: str = 'model.pt', 
                 loss_fn: torch.nn.modules.loss._Loss = MSELoss(), device: torch.device = torch.device('cpu'), 
                 epochs: int = 100, patience: int = 10, logger: logging.Logger = logging.getLogger("Trainer"), 
                 wandb: object = None) -> None:
        self.model: torch.nn.Module = model
        self.data_generators: Dict = data_generators
        self.optim: str = optim
        self.optim_params: Dict = optim_params
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = lr_scheduler
        self.model_save_dir: str = model_save_dir
        self.model_name: str = model_name
        self.loss_fn: torch.nn.modules.loss._Loss = loss_fn
        self.device: torch.device = device
        self.epochs: int = epochs
        self.patience: int = patience
        self.logger: logging.Logger = logger
        self.logger.setLevel(logging.INFO)
        self.wandb: object = wandb
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        if self.model_save_dir is not None and not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self._register_wandb_params()
        self._validate_inputs()
        self._set_optimizer()
    
    def _assign_device_to_data(self, phi: torch.Tensor, y: torch.Tensor, 
                           graph_features: torch.Tensor = None, edge_indices: torch.Tensor = None) -> Tuple:
        """
        Assign the device to the features and the target
        """
        # Ensure phi is on the correct device
        phi, y = phi.to(self.device), y.to(self.device)
        res = (phi, y)
        
        if graph_features is not None:
            graph_features = graph_features.to(self.device)
            res += (graph_features,)
        if edge_indices is not None:
            edge_indices = edge_indices.to(self.device)
            res += (edge_indices,)
        
        return res

    
    @abstractmethod
    def _validate_inputs(self) -> None:
        """
        Validate the inputs to the trainer
        """

        print(f"Base classes of StepLR: {StepLR.__bases__}")
        print(f"Is _LRScheduler among the bases? {LRScheduler in StepLR.__bases__}")

        if self.lr_scheduler is not None:
            print(f"Type of lr_scheduler in Trainer: {type(self.lr_scheduler)}")
            print(f"Is instance of _LRScheduler? {isinstance(self.lr_scheduler, LRScheduler)}")
            if not isinstance(self.lr_scheduler, LRScheduler):
                raise TypeError("lr_scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler")

        if not isinstance(self.data_generators, Dict):
            raise TypeError("data_generators must be a dictionary.")
        if not set(self.data_generators.keys()).issubset({TRAIN, VAL, TEST}):
            raise ValueError("The keys of data_generators must be a subset of {\'train\', \'val\', and \'test\'}")
        if self.optim not in OPTIMIZERS:
            raise TypeError("The optimizer must be one of the following: {}".format(OPTIMIZERS.keys()))
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, LRScheduler):
            raise TypeError("lr_scheduler must be an instance of torch.optim.lr_scheduler")
        if not isinstance(self.loss_fn, torch.nn.modules.loss._Loss):
            raise TypeError("loss_fn must be an instance of torch.nn.modules.loss._Loss")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be an instance of torch.device")
        if not isinstance(self.logger, logging.Logger):
            raise TypeError("logger must be an instance of logging.Logger")
    
    def _set_optimizer(self) -> None:
        """
        Set the optimizer for the trainer
        """
        self.optimizer = OPTIMIZERS[self.optim](self.model.parameters(), **self.optim_params)
    
    def _register_wandb_params(self) -> None:
        """
        Register the parameters for wandb
        """
        if self.wandb is not None:
            self.wandb.config.update({
                "optimizer": self.optim,
                "optimizer_params": self.optim_params,
                "loss_fn": self.loss_fn,
                "epochs": self.epochs,
                "patience": self.patience
            }, allow_val_change=True)

    def _set_optimizer(self) -> None:
        """
        Set the optimizer and initialize the learning rate scheduler if provided
        """
        # Initialize the optimizer
        self.optimizer = OPTIMIZERS[self.optim](self.model.parameters(), **self.optim_params)

        # Initialize the scheduler (if provided)
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, LRScheduler):
                # Scheduler already initialized; use it directly
                pass
            elif callable(self.lr_scheduler):
                # Scheduler is a class (e.g., StepLR); initialize it
                self.lr_scheduler = self.lr_scheduler(self.optimizer, **self.optim_params.get("scheduler_params", {}))
            else:
                raise TypeError("lr_scheduler must be an instance of or a callable subclass of torch.optim.lr_scheduler.LRScheduler")

    
    @abstractmethod
    def train(self) -> None:
        """
        Model training
        """
        pass

    @abstractmethod
    def predict(self) -> None:
        """
        Model evaluation
        """
        pass

class Trainer(BaseTrainer):
    """
    Trainer for the nonlinear spatial causal inference model

    Additional Arguments
    --------------
    window_size: int, the window size for the treatment/intervention map
    t_idx: int, the index of the treatment for which we want to estimate the causal effect
    gps_model: GeneralizedPropensityScoreModel, the model for estimating the propensity score
    sw_model: scipy.stats.gaussian_kde, the model for estimating the stabilized weights
    """
    def __init__(self, model: torch.nn.Module, data_generators: Dict, optim: str, optim_params: Dict, window_size: int,
                 t_idx: int = 0, lr_scheduler: torch.optim.lr_scheduler = None, model_save_dir: str = None, 
                 model_name: str = 'model.pt', loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), M_train: torch.Tensor = None, M_val: torch.Tensor = None,
                 gps_model: GeneralizedPropensityScoreModel = None, sw_model: scipy.stats._kde.gaussian_kde = None, 
                 device: torch.device = torch.device('cpu'), epochs: int = 100, patience: int = 10, 
                 logger: logging.Logger = logging.getLogger("Trainer"), wandb: object = None) -> None:
        self.M_train = M_train # add M_train as input
        self.M_val = M_val # add M_val as input
        self.window_size: int = window_size
        self.t_idx: int = t_idx
        self.gps_model: GeneralizedPropensityScoreModel = gps_model
        self.sw_model: scipy.stats._kde.gaussian_kde = sw_model

        super(Trainer, self).__init__(model, data_generators, optim, optim_params, lr_scheduler, model_save_dir, model_name, 
                                      loss_fn, device, epochs, patience, logger, wandb)
        self._validate_inputs()
        self._set_optimizer()
    
    def _validate_inputs(self) -> None:
        # if not isinstance(self.model, (LinearSCI, NonlinearSCI)):
        #     raise TypeError("model must be an instance of LinearSCI or NonlinearSCI.")"
        if self.gps_model is not None and not isinstance(self.gps_model, GeneralizedPropensityScoreModel):
            raise TypeError("gps_model must be an instance of GeneralizedPropensityScoreModel.")
        if self.sw_model is not None and not isinstance(self.sw_model, scipy.stats._kde.gaussian_kde):
            raise TypeError("sw_model must be an instance of scipy.stats._kde.gaussian_kde.")
        super(Trainer, self)._validate_inputs()
    
    def _train_step(self) -> Tuple:
        """
        Perform a single training step
        """
        self.model.to(self.device)
        self.model.train()
        y_train_pred = torch.empty(0).to(self.device)
        y_train_true = torch.empty(0).to(self.device)

        for step, batch in enumerate(self.data_generators[TRAIN]):
            samples = self._assign_device_to_data(*batch)
            
            # Dynamically determine batch size from current batch
            # current_batch_size = samples[0].size(0)
            # batch_indices = torch.arange(step * current_batch_size, (step + 1) * current_batch_size)

            phi = samples[0]
            y = samples[1].view(-1)
            self.optimizer.zero_grad()
            y_pred = self.model(phi).float().squeeze()
            assert y_pred.shape == y.shape, "The shape of the prediction must be the same as the target"
            
            # Extract the batch-specific mask from M_train
            batch_size = y.size(0)
            batch_start = step * batch_size
            batch_M = self.M_train[batch_start:batch_start + batch_size]

            # Compute Batch Loss
            loss = self.loss_fn(y_pred, y.float(), M=batch_M)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record the predictions
            y_train_pred = torch.cat((y_train_pred, y_pred), dim=0)
            y_train_true = torch.cat((y_train_true, y), dim=0)

        # Compute overall Train Loss
        train_loss = self.loss_fn(y_train_pred, y_train_true, M=self.M_train).item()
        return train_loss, step
    
    def train(self) -> None:
        best_loss = 1e9
        best_model_state_dict = None
        trigger_times = 0
        
        self.logger.info("Training started:\n")
        for epoch in range(self.epochs):
            # Training
            train_start = time.time()
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            self.logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:7.6f}")
            # Log the training time and loss
            train_loss, step = self._train_step()
            train_time = time.time() - train_start
            self.logger.info("{:.0f}s for {} steps - {:.0f}ms/step - loss {:.4f}" \
                  .format(train_time, step + 1, train_time * 1000 // (step + 1), train_loss))
            # Validation
            val_start = time.time()
            self.logger.info("Validation:")
            val_loss = self.validate()
            val_time = time.time() - val_start
            self.logger.info("{:.0f}s - loss {:.4f}\n".format(val_time, val_loss))
            if self.wandb is not None:
                self.wandb.log({"train_loss": train_loss, "validation_loss": val_loss})
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Early stopping
            if val_loss > best_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    # Trigger early stopping and save the best model
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

        if trigger_times < self.patience:
            self.logger.info("Training completed without early stopping.")
    
    def validate(self) -> float:
        """
        Evaluate the model on the validation data using only basis functions (phi) and target values (y).
        """
        y_val_pred = torch.empty(0).to(self.device)
        y_val_true = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[VAL]:
                samples = self._assign_device_to_data(*batch)
                phi, y = samples[0], samples[1].view(-1)
                y_pred = self.model(phi).float().squeeze()
                y_val_pred = torch.cat((y_val_pred, y_pred), dim=0)
                y_val_true = torch.cat((y_val_true, y), dim=0)

        assert y_pred.shape == y.shape, "The shape of the prediction must be the same as the target"
        # Pass M_val to NewLoss for compatibility
        val_loss = self.loss_fn(y_val_pred, y_val_true, M=self.M_val).item()
        return val_loss
    
    def predict(self) -> np.ndarray:
        """
        Evaluate the model on the test data using only basis functions (phi).
        """
        y_test_pred = torch.empty(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in self.data_generators[TEST]:
                phi, _ = self._assign_device_to_data(*batch)
                y_pred = self.model(phi).float()
                y_test_pred = torch.cat((y_test_pred, y_pred), dim=0)

        return y_test_pred.cpu().numpy()
