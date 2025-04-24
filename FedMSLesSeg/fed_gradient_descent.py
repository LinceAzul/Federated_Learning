import torch

def gradient_descent(w, b, X, y, grad_w, grad_b):
    z = X @ w + b
    y_pred = torch.sigmoid(z)

    dL_dz = y_pred - y
    aux_grad_w = (X.T @ dL_dz) / len(X)
    aux_grad_b = dL_dz.mean(dim=0)

    return grad_w + aux_grad_w, grad_b + aux_grad_b

def initialize_weights(input_dim):
    w = torch.zeros((input_dim, 1), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)
    return w, b

def compute_client_gradients(client_loader, w, b):
    grad_w, grad_b = torch.zeros_like(w), torch.zeros_like(b)
    for X, y in client_loader:
        grad_w, grad_b = gradient_descent(w, b, X, y, grad_w, grad_b)
        return grad_w, grad_b

def aggregate_gradients(clients_loaders, w, b):
    grads_w, grads_b = [], []

    for client_loader in clients_loaders:
        grad_w, grad_b = compute_client_gradients(client_loader, w, b)
        grads_w.append(grad_w)
        grads_b.append(grad_b)

    avg_grad_w = torch.stack(grads_w).mean(dim=0)
    avg_grad_b = torch.stack(grads_b).mean(dim=0)
    return avg_grad_w, avg_grad_b

def train_federated_model(clients_datasets, input_dim, lr=0.01, epochs=10000):
    w, b = initialize_weights(input_dim)
    client_loaders = [torch.utils.data.DataLoader(client_data[0], batch_size=len(client_data[0])) for client_data in clients_datasets]

    for _ in range(epochs):
        avg_grad_w, avg_grad_b = aggregate_gradients(client_loaders, w, b)
        w -= lr * avg_grad_w
        b -= lr * avg_grad_b

    return w, b

from FedMSLesSeg.data_loader import load_and_prepare_data, split_dataset, split_dataset_among_clients
from FedMSLesSeg.gradient_descent import evaluate_model, train_logistic_regression
from FedMSLesSeg.fed_gradient_descent import train_federated_model
import matplotlib.pyplot as plt
# from FedMSLesSeg.logger import get_logger, configure_logging
import time

def main(i=1):
    # Load and preprocess the dataset
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_data()
    print("Dataset loaded successfully.")

    # Split dataset for global evaluation
    global_train, global_test = split_dataset_among_clients([dataset])[0]
    fed_avg_losses, fed_avg_errors, fed_times = [], [], []
    central_avg_losses, central_avg_errors, central_global_times = [], [], []
    
    
    clients_datasets = split_dataset_among_clients(split_dataset(dataset, split_ratios=i))

    # Determine input dimension from a sample
    X_sample, _ = clients_datasets[0][0][0]
    input_dim = X_sample.shape[0]

    # === Federated Training ===
    print("\nStarting Federated Training...")
    start_time = time.time()
    w_fed, b_fed = train_federated_model(clients_datasets, input_dim)
    fed_training_time = time.time() - start_time
    print(f"Federated training completed in {fed_training_time:.2f} seconds.")
    
    # Evaluate federated model
    print("Evaluating Federated Model...")
    avg_loss, avg_error = evaluate_model(w_fed, b_fed, global_test)
    fed_avg_losses.append((i, avg_loss))
    fed_avg_errors.append((i,avg_error))
    fed_times.append((i,fed_training_time))
    print(f"Final Test Loss: {avg_loss:.4f} - Final Test Error: {avg_error:.4f}")
        
    print("Centralized training times:", central_global_times)
    print("Federated training timess:", fed_times)
    print("Centralized average losses:", central_avg_losses)
    print("Centralized average errors:", central_avg_errors)
    print("Federated average losses:", fed_avg_losses)
    print("Federated average errors:", fed_avg_errors)

    return fed_avg_losses, fed_avg_errors, fed_times, central_avg_losses, central_avg_errors, central_global_times

if __name__ == "__main__":
    main(36)
