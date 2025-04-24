from torch.utils.data import TensorDataset, random_split
from sklearn.datasets import load_breast_cancer
import torch

def load_and_prepare_data():
    data = load_breast_cancer()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1) 
    return TensorDataset(X, y)

import numpy as np

def get_split_ratios(n, min_ratio=0.01):
    # Proporción básica de cada división
    base = 1 / n
    
    # Si la proporción es menor que el mínimo permitido, ajustamos
    if base < min_ratio:
        base = min_ratio
    
    # Crear la lista de proporciones
    ratios = [base] * n
    total = sum(ratios)  # Sumar todas las proporciones

    # Calcular la diferencia para ajustar la última proporción
    diff = round(1.0 - total, 2)
    
    # Ajustar la última proporción
    ratios[-1] += diff

    return ratios

def split_dataset(dataset, split_ratios=100):
    split_ratios = get_split_ratios(split_ratios)  # Obtener las proporciones
    
    total_size = len(dataset)  # Tamaño total del dataset
    sizes = [int(r * total_size) for r in split_ratios[:-1]]  # Tamaños de cada división, excepto la última
    sizes.append(total_size - sum(sizes))  # Ajustar el tamaño de la última división

    # Dividir el dataset
    clients_dataset = random_split(dataset, sizes)

    return clients_dataset

def split_dataset_among_clients(clients_dataset, train_ratio=0.8):
    clients_dataset_formated = []

    for client_dataset in clients_dataset:
        train_size = int(train_ratio * len(client_dataset))
        val_size = len(client_dataset) - train_size
        clients_dataset_formated.append(random_split(client_dataset, [train_size, val_size]))

    return clients_dataset_formated