# deep_ps

- custom_dk_imports.py: Imports certain required classes from deep_sci (DeepKrigingMLP, etc.)
- ps_synthetic: Compares fit of basic GP and custom GP (with new loss = mse + binary cross entropy loss).
Uses 50x50 synthetic data grid.
- ps_synthetic_DK (Work in Progress): Compares fit of basic GP and DeepKriging (with MSE loss).
Uses 50x50 synthetic data grid. No covariates, only takes in spatial coords.
- DK_troubleshooting (Most Up-to-Date): Attempts to improve MSE of DeepKriging compared to basic GP.
Uses 50x50 synthetic data grid. No covariates, only takes in spatial coords.
