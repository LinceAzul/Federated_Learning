
from torch.utils.data import DataLoader
import torch

def train_logistic_regression(train, lr=0.01, epochs=10000):
    dataloader = DataLoader(train, batch_size=len(train), shuffle=True)
    X_batch, y_batch = next(iter(dataloader))

    N, D = X_batch.shape
    w = torch.zeros((D, 1), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)

    for _ in range(epochs):
        grad_w = torch.zeros_like(w)
        grad_b = torch.zeros_like(b)
        for X_batch, y_batch in dataloader:
            z = X_batch @ w + b
            y_pred = torch.sigmoid(z)

            dL_dz = y_pred - y_batch
            grad_w += (X_batch.T @ dL_dz) / N
            grad_b += dL_dz.mean(dim=0)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b

def evaluate_model(w, b, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    total_loss = 0.0
    total_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            z_test = X_test @ w + b
            y_test_pred = torch.sigmoid(z_test)

            loss = - (y_test * torch.log(y_test_pred + 1e-7) + (1 - y_test) * torch.log(1 - y_test_pred + 1e-7)).sum().item()
            y_pred_labels = (y_test_pred >= 0.5).float()
            error = (y_pred_labels != y_test).float().sum().item()

            total_loss += loss
            total_error += error
            total_samples += y_test.size(0)

    avg_loss = total_loss / total_samples
    avg_error = total_error / total_samples

    return avg_loss, avg_error