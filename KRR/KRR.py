import torch
import pickle
import os

from sklearn.model_selection import train_test_split

# Import required libraries
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import numpy as np

def load_data(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def prepare_dataset(loaded_data):
    """
    Prepare the dataset for training and testing.
    
    Parameters:
    -----------
    loaded_data : dict
        Dictionary containing features and labels
    
    Returns:
    --------
    tuple
        Tuple containing training and testing data
    """
    all_features = loaded_data["features"]
    all_labels = loaded_data["labels"]
    
    print(f"Features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    np.random.seed(666)
    
    sample_size = 10000
    indices = np.random.choice(len(all_features), sample_size, replace=False)
    sampled_features = all_features[indices]
    sampled_labels = all_labels[indices]

    x_train, x_test, y_train, y_test = train_test_split(
        sampled_features, sampled_labels, test_size=0.2, random_state=666)

    # Convert PyTorch tensors to NumPy arrays for scikit-learn
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    x_test_np = x_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Print data shapes for confirmation
    print(f"Training data shapes: x={x_train_np.shape}, y={y_train_np.shape}")
    print(f"Testing data shapes: x={x_test_np.shape}, y={y_test_np.shape}")
    
    return x_train_np, y_train_np, x_test_np, y_test_np

def grid_search_kernel_ridge(x_train_np, y_train_np, x_test_np, y_test_np, 
                             alphas=[0.001, 0.01, 0.1, 1.0], 
                             gammas=[0.0001, 0.001, 0.01, 0.1],
                             test_dims=10, 
                             kernel='rbf', 
                             verbose=True):
    """
    Perform grid search for KernelRidge hyperparameters.
    
    Parameters:
    -----------
    x_train_np : array-like
        Training features
    y_train_np : array-like
        Training targets
    x_test_np : array-like
        Test features
    y_test_np : array-like
        Test targets
    alphas : list, default=[0.001, 0.01, 0.1, 1.0]
        List of alpha values to try
    gammas : list, default=[0.0001, 0.001, 0.01, 0.1]
        List of gamma values to try
    test_dims : int, default=10
        Number of output dimensions to use for evaluation
    kernel : str, default='rbf'
        Kernel type to use
    verbose : bool, default=True
        Whether to print progress
        
    Returns:
    --------
    dict
        Dictionary containing best parameters and score
    """
    best_score = -float('inf')
    best_params = {}
    
    # Limit test dimensions to available dimensions
    test_dims = min(test_dims, y_train_np.shape[1])
    
    # Simple grid search
    for alpha in alphas:
        for gamma in gammas:
            if verbose:
                print(f"Trying parameters: alpha={alpha}, gamma={gamma}")
            
            base_model = KernelRidge(
                alpha=alpha,
                kernel=kernel,
                gamma=gamma
            )
            
            model = MultiOutputRegressor(base_model, n_jobs=-1)
            model.fit(x_train_np, y_train_np[:, :test_dims])
            score = model.score(x_test_np, y_test_np[:, :test_dims])
            
            if verbose:
                print(f"Average R² score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = {'alpha': alpha, 'gamma': gamma}
    
    if verbose:
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score
    }

if __name__ == "__main__":
    # Load data
    data_dir = "./dataset.pkl"
    loaded_data = load_data(data_dir)
    
    # Prepare dataset
    x_train_np, y_train_np, x_test_np, y_test_np = prepare_dataset(loaded_data)
    
    # Perform grid search for Kernel Ridge regression
    results = grid_search_kernel_ridge(x_train_np, y_train_np, x_test_np, y_test_np)
    
    print("best_params:", results['best_params'])
    print("best_score:", results['best_score'])
    
    # Save results to a file
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Results saved to results.pkl")
    
    # use the best parameters to train the final model
    best_alpha = results['best_params']['alpha']
    best_gamma = results['best_params']['gamma']

    print(f"Training final model with alpha={best_alpha}, gamma=    {best_gamma}")

    
    final_base_model = KernelRidge(
        alpha=best_alpha,
        kernel='rbf',
        gamma=best_gamma
    )

    final_model = MultiOutputRegressor(final_base_model, n_jobs=-1)
    final_model.fit(x_train_np, y_train_np)

    final_score = final_model.score(x_test_np, y_test_np)
    print(f"Final model R² score: {final_score:.4f}")

    model_path = 'best_krr_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)

    print(f"Best model saved to {model_path}")