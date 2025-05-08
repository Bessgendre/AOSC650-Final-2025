import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from e3nn import o3

import os
import numpy as np

import pickle

from tools.AtomAtom import AtomBlockReconstructor
from e3nn_model import EquivariantModel

import time
import datetime

def load_data(path):
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    
    print("Dataset Size:", dataset["features"].shape)
    return dataset

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

    return x_train, y_train, x_test, y_test

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to torch tensors here for efficiency
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])

def worker_init_fn(worker_id):
    # Set unique random seed for each worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == "__main__":
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Print memory info
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    # Set PyTorch to use deterministic algorithms for reproducibility
    # Note: This may slow down training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Setting to True can speed up training if input sizes don't change
    
    # Load data
    path = "./dataset.pkl"
    dataset = load_data(path)

    x_train, y_train, x_test, y_test = prepare_dataset(dataset)
    print(f"Train set: {x_train.shape}")
    print(f"Test set: {x_test.shape}")
    
    # Prepare loader with multiple workers for faster data loading
    num_workers = 4 if torch.cuda.is_available() else 0  # Use multiple workers on GPU
    pin_memory = torch.cuda.is_available()  # Pin memory for faster data transfer to GPU
    
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Get the irreps
    carbon_reconstructor = AtomBlockReconstructor("2x0e+1x1o", "2x0e+1x1o")
    irreps_structure = carbon_reconstructor.all_decomposed_irreps
    irreps = o3.Irreps("+".join(["+".join(sublist) for sublist in irreps_structure]))

    irreps_in = irreps
    irreps_hidden = irreps_in * 2
    irreps_out = irreps
    
    # Create model
    model = EquivariantModel(irreps_in, irreps_hidden, irreps_out)
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Move model to device
    model = model.to(device)
    
    # Print the model size
    total_weights = sum(p.numel() for p in model.parameters())
    print(f"Total number of weights in model: {total_weights}")
    
    # Generate timestamp for folder naming
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"./tf-logs/run_{timestamp}"
    model_dir = os.path.join(save_dir, "models")
    log_dir = os.path.join(save_dir, "logs")

    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler options:
    # Step decay - reduces lr by gamma every step_size epochs
    scheduler_step = StepLR(optimizer, step_size=10, gamma=0.5)  # Halves the learning rate every 20 epochs

    # Reduce on plateau - reduces lr when validation loss plateaus
    scheduler_plateau = ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Reduce LR when monitored quantity stops decreasing
        factor=0.5,           # Multiply lr by this factor
        patience=5,           # Number of epochs with no improvement after which LR will be reduced
        verbose=True,         # Print message when LR is reduced
        min_lr=1e-6           # Lower bound on the learning rate
    )

    # Choose which scheduler to use (you can use both if you want)
    use_step_scheduler = True
    use_plateau_scheduler = True

    # Initialize TensorBoard writer with timestamped log directory
    writer = SummaryWriter(log_dir=log_dir)

    # Variables for tracking the best model
    best_val_loss = float('inf')
    best_epoch = 0

    # Training and validation loop
    num_epochs = 100
    start_time = time.time()  # Record start time

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Record epoch start time
        
        # Empty cache before each epoch to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Training loop
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            # Move data to device
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                # Move data to device
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()

        val_loss /= len(test_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Update learning rate
        if use_step_scheduler:
            scheduler_step.step()
        if use_plateau_scheduler:
            scheduler_plateau.step(val_loss)

        # Synchronize CUDA for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        writer.add_scalar("Time/Epoch", epoch_time, epoch)
        
        # GPU memory usage
        if torch.cuda.is_available():
            writer.add_scalar("GPU/Memory_Allocated", torch.cuda.memory_allocated(0) / 1e9, epoch)
            writer.add_scalar("GPU/Memory_Reserved", torch.cuda.memory_reserved(0) / 1e9, epoch)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            # Save model state
            model_to_save = model.module if hasattr(model, 'module') else model  # Handle DataParallel
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_step_state_dict': scheduler_step.state_dict() if use_step_scheduler else None,
                'scheduler_plateau_state_dict': scheduler_plateau.state_dict() if use_plateau_scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
            }, os.path.join(model_dir, 'best_model.pth'))
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s (New Best Model)")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")

        # Save a checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Save model state
            model_to_save = model.module if hasattr(model, 'module') else model  # Handle DataParallel
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_step_state_dict': scheduler_step.state_dict() if use_step_scheduler else None,
                'scheduler_plateau_state_dict': scheduler_plateau.state_dict() if use_plateau_scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
            }, os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Synchronize CUDA before calculating total time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
    print(f"All files saved in directory: {save_dir}")

    # Add final metrics to TensorBoard
    writer.add_scalar("Best/Epoch", best_epoch, 0)
    writer.add_scalar("Best/ValidationLoss", best_val_loss, 0)
    writer.add_scalar("Time/Total", total_time, 0)

    # Close the TensorBoard writer
    writer.close()