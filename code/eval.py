import torch
from dataloader import get_test_dataloader, get_train_dataloader
from model import SimpleModel
from tqdm import tqdm
from util import dbg

def evaluate_model(model, dataloader, device=None):
    model.eval()

    diff = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).to(torch.float32)
            labels_degrees = torch.argmax(labels, dim=1).to(torch.float32)
            # dbg(preds, labels_degrees)
            batch_diff = torch.mean(torch.abs(preds - labels_degrees))
            diff.append(batch_diff.item())

    avg_diff = torch.mean(torch.tensor(diff))
    return avg_diff


def generate_histogram_vals(model, dataloader, device=None):
    model.eval()

    all_deg_diff = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).to(torch.float32)
            labels_degs = torch.argmax(labels, dim=1).to(torch.float32)
            deg_diffs = torch.abs(preds - labels_degs)
            all_deg_diff.extend(deg_diffs.cpu().numpy())

    return all_deg_diff

if __name__ == "__main__":
    model = SimpleModel()
    model.load_state_dict(torch.load("../model_weights/simple_model.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    train_dataloader = get_train_dataloader(num_images=512, image_size=(64, 64), batch_size=64, shuffle=True)
    test_dataloader = get_test_dataloader(num_images=512, image_size=(64, 64), batch_size=64, shuffle=False)
    
    train_degree_diff = evaluate_model(model, train_dataloader, device)
    print(f"Train Average Degree Difference: {train_degree_diff.item():.2f}")

    test_degree_diff = evaluate_model(model, test_dataloader, device)
    print(f"Test Average Degree Difference: {test_degree_diff.item():.2f}")