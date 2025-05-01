This is a boilerplate template for developing and deploying PyTorch models in a production environment.

## Features
- Modular training code
- BERT-like model inference
- Docker support
- CLI support for hyperparameters

## Files
- `train.py`: Train your PyTorch model with CLI arguments
- `model.py`: Define your neural network architecture
- `utils.py`: Utility functions for data loading, saving, and reproducibility
- `inference.py`: Load and predict using a saved model
- `Dockerfile`: Containerize the entire setup

## Getting Started

### Train the Model
```bash
python train.py --epochs 10 --lr 0.001 --input_dim 100 --output_dim 10
```

### Run Inference
```bash
python inference.py --model_path checkpoints/model.pt
```

### Build Docker Container
```bash
docker build -t pytorch-sde .
docker run --rm pytorch-sde
