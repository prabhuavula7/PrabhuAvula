import torch
from model import SimpleNet
import argparse

def load_model(path, input_dim, output_dim):
    model = SimpleNet(input_dim, output_dim)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1)
    return pred.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--input_dim", type=int, default=100)
    parser.add_argument("--output_dim", type=int, default=10)
    args = parser.parse_args()

    sample = torch.randn(1, args.input_dim)
    model = load_model(args.model_path, args.input_dim, args.output_dim)
    result = predict(model, sample)
    print(f"Prediction: {result}")
