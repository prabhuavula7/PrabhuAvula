import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimpleNet
from utils import get_dataloaders, save_model, set_seed
import argparse
import os

def train(config):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(config)

    model = SimpleNet(config.input_dim, config.output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {total_loss/len(train_loader):.4f}")

    save_model(model, config.output_dir, config.model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=100)
    parser.add_argument("--output_dim", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--model_name", type=str, default="model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
