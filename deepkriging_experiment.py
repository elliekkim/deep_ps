import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from itertools import product
from custom_dk_imports import DeepKrigingMLP, Trainer, NewLoss
import yaml

# --------------------------- Helper Classes ---------------------------

class DeepKrigingEmbedding3d(torch.nn.Module):
    def __init__(self, K: int):
        """
        DeepKriging 3D Embedding Layer
        :param K: Number of basis resolutions
        """
        super(DeepKrigingEmbedding3d, self).__init__()
        self.K = K
        self.num_basis = [(9 * 2 ** (h - 1) + 1) ** 2 for h in range(1, self.K + 1)]
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

# --------------------------- Helper Functions ---------------------------

def load_and_preprocess(data_path, icar_ps_path, crop_size=None):
    """
    Load and optionally crop the data, and preprocess the grid data.
    Updates unobserved cells in the original data using ICAR+PS predictions.
    """
    # Load the main data
    data = np.load(data_path)
    icar_ps_data = np.load(icar_ps_path)

    # If crop_size is provided, crop the center region
    if crop_size:
        start_y = (data.shape[0] - crop_size) // 2
        start_x = (data.shape[1] - crop_size) // 2
        data = data[start_y:start_y + crop_size, start_x:start_x + crop_size]
        icar_ps_data = icar_ps_data[start_y:start_y + crop_size, start_x:start_x + crop_size]

        print(f"Cropped data to shape: {data.shape}")
        print(f"Cropped ICAR+PS data to shape: {icar_ps_data.shape}")
    else:
        print(f"Using full data shape: {data.shape}")
        print(f"Using full ICAR+PS data shape: {icar_ps_data.shape}")

    # Process observed and unobserved cells
    all_indices = list(product(np.arange(data.shape[0]), np.arange(data.shape[1])))
    observed_indices = [(y, x) for y, x in all_indices if not np.isnan(data[y, x])]
    unobserved_indices = [(y, x) for y, x in all_indices if np.isnan(data[y, x])]

    # Initialize DataFrame
    df = pd.DataFrame(all_indices, columns=['grid_row', 'grid_col'])
    df['temp_avg'] = np.nan

    # Update observed points
    for y, x in observed_indices:
        df.loc[(df['grid_row'] == y) & (df['grid_col'] == x), 'temp_avg'] = data[y, x]

    # Update unobserved points with ALL ZEROES FOR NOW (We can't fill this in with ICAR+PS data anymore bc the indices won't line up...)
    for y, x in unobserved_indices:
        df.loc[(df['grid_row'] == y) & (df['grid_col'] == x), 'temp_avg'] = np.nan

    print(f"Processed data: {df['temp_avg'].notna().sum()} non-NaN values.")
    return df, observed_indices

def prepare_tensors(df):
    """Prepare tensors for training by normalizing grid_points and temp_values."""
    grid_points = df[['grid_row', 'grid_col']].to_numpy()
    temp_values = df['temp_avg'].to_numpy()

    temp_values = (temp_values - temp_values.min()) / (temp_values.max() - temp_values.min())

    # Save min and max for normalization
    grid_points_min = grid_points.min(axis=0)  # [min_y, min_x]
    grid_points_max = grid_points.max(axis=0)  # [max_y, max_x]

    # Normalize grid_points
    grid_points = (grid_points - grid_points.min(axis=0)) / (grid_points.max(axis=0) - grid_points.min(axis=0))

    return grid_points, temp_values, grid_points_min, grid_points_max

def create_data_loaders(observed_indices, grid_points, temp_values, K, batch_size=32):
    """Create train, validation, and test loaders using DeepKrigingEmbedding."""
    observed_indices_set = set(observed_indices)
    M = np.zeros(len(grid_points))

    # Mark observed points in M
    for i in range(len(grid_points)):
        if tuple(grid_points[i]) in observed_indices_set:
            M[i] = 1
    
    s_train, s_val, y_train, y_val, M_train, M_val = train_test_split(grid_points, temp_values, M, test_size=0.2, random_state=42)

    M_train = torch.tensor(M_train, dtype=torch.float32)
    M_val = torch.tensor(M_val, dtype=torch.float32)
    s_train_tensor = torch.tensor(s_train, dtype=torch.float32)
    s_val_tensor = torch.tensor(s_val, dtype=torch.float32)
    s_all_tensor = torch.tensor(grid_points, dtype=torch.float32)

    embedding_layer = DeepKrigingEmbedding3d(K)

    # Apply embedding
    embedding_layer.eval()
    with torch.no_grad():
        phi_train = embedding_layer(s_train_tensor)
        phi_val = embedding_layer(s_val_tensor)
        phi_all = embedding_layer(s_all_tensor)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(phi_train, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(phi_val, y_val_tensor), batch_size=batch_size, shuffle=False)

    input_dim = sum(embedding_layer.num_basis)  # Calculate dynamically from embedding layer
    return train_loader, val_loader, input_dim, phi_all, M_train, M_val

def normalize_observed_indices(observed_indices, grid_points_min, grid_points_max):
    """
    Normalize observed_indices using the same min-max scaling as grid_points.
    """
    observed_indices_normalized = [
        (
            (y - grid_points_min[0]) / (grid_points_max[0] - grid_points_min[0]),
            (x - grid_points_min[1]) / (grid_points_max[1] - grid_points_min[1])
        )
        for y, x in observed_indices
    ]
    return observed_indices_normalized

def train_model(train_loader, val_loader, grid_points, grid_points_min, grid_points_max, M_train, M_val, observed_indices, temp_values, loss_type, input_dim, K, epochs=100):
    """Train the DeepKriging model using the embedding layer."""

    observed_indices = [(y, x) for y, x in observed_indices]
    observed_indices_normalized = normalize_observed_indices(observed_indices, grid_points_min, grid_points_max)

    grid_points_tuples = [tuple(pt) for pt in grid_points]
    grid_point_idx_dict = {pt: idx for idx, pt in enumerate(grid_points_tuples)}
    observed_grid_cell_indices_flat = []

    for i in observed_indices_normalized:
        observed_grid_cell_indices_flat.append(grid_point_idx_dict[i])
    observed_grid_cell_indices_flat = np.array(observed_grid_cell_indices_flat)

    model = DeepKrigingMLP(input_dim, num_hidden_layers=1, hidden_dims=100, K=K, activation='relu')


    loss_fn = NewLoss(s_all=grid_points, observed_indices=observed_grid_cell_indices_flat, y_all=temp_values) if loss_type == 'newloss' else torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(
        model=model,
        data_generators={'train': train_loader, 'val': val_loader},
        optim='adam',
        optim_params={'lr': 1e-3},
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        M_train=M_train,
        M_val=M_val,
        window_size=10,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        epochs=epochs,
        patience=10,
    )
    trainer.train()
    return model

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# --------------------------- Main Execution ---------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepKriging Experiment Runner")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Print configuration for logging
    print("Loaded Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # Load, crop, and preprocess data
    df, observed_indices = load_and_preprocess(config['data_path'], config['icar_ps_path'], config.get('crop_size'))

    # Prepare tensors and data loaders
    grid_points, temp_values, grid_points_min, grid_points_max = prepare_tensors(df)

    # Print number of Nan values in temp_values:
    print("Number of NaN values in temp_values: ", np.isnan(temp_values).sum())
    print("Number of zeroes (unobserved) vals in temp_values: ", (temp_values == 0).sum())

    train_loader, val_loader, input_dim, phi_all, M_train, M_val = create_data_loaders(observed_indices, grid_points, temp_values, config['K'], batch_size=config['batch_size'])

    # Train the model
    deepkriging_model = train_model(train_loader, val_loader, grid_points, grid_points_min, grid_points_max, M_train, M_val, observed_indices, temp_values, config['loss_type'], input_dim, config['K'], config['epochs'])
    print("Model training complete!")

    # Evaluate the model
    print("Run the model on the data...")
    deepkriging_model.eval()

    with torch.no_grad():
        y_pred_deepkriging = deepkriging_model(phi_all).cpu().numpy()

    # First, plot histogram of DK predictions
    plt.hist(y_pred_deepkriging, bins=50, alpha=0.7)
    plt.title("Distribution of Predictions (DeepKriging)")
    plt.xlabel("Prediction")
    plt.ylabel("Frequency")
    plt.show()

    # Dynamically determine grid dimensions
    cells_y, cells_x = int(df['grid_row'].max()) + 1, int(df['grid_col'].max()) + 1  # Rows and columns

    # Then, compare to a plot of the True Process:
    plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(temp_values.reshape(cells_y, cells_x), cmap='viridis', aspect='equal')
    # plt.title('True Values')
    # plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(y_pred_deepkriging.reshape(cells_y, cells_x), cmap='viridis', aspect='equal')
    plt.title('DeepKriging Predictions')
    plt.colorbar()

    # Save the figure to a file
    if config.get('crop_size') != None:
        output_filename = f"true_vs_dk_k{config['K']}_crop{config['crop_size']}.png"
    else:
        output_filename = f"true_vs_dk_k{config['K']}_notcrop.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Figure saved to {output_filename}")

    plt.show()

    print("DeepKriging Experiment Complete!")

if __name__ == "__main__":
    main()
