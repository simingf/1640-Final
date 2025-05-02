import torch
from dataloader import get_test_dataloader, get_train_dataloader
from model import SimpleModel
from tqdm import tqdm
from util import dbg

def evaluate_model(model, weights_path, dataloader, device=None):
    model = model().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    diff = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            labels_degrees = torch.argmax(labels, dim=1)
            dbg(preds, labels_degrees)
            batch_diff = torch.mean(torch.abs(preds - labels_degrees))
            diff.append(batch_diff.item())

    avg_diff = torch.mean(torch.tensor(diff))
    return avg_diff


if __name__ == "__main__":
    model = SimpleModel
    weights_path = "../model_weights/simple_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    train_dataloader = get_train_dataloader(num_images=512, image_size=(64, 64), batch_size=64, shuffle=True)
    test_dataloader = get_test_dataloader(num_images=512, image_size=(64, 64), batch_size=64, shuffle=False)
    
    accuracy = evaluate_model(model, weights_path, train_dataloader, device)
    print(f"Train Accuracy: {accuracy.item() * 100:.2f}%")

    accuracy = evaluate_model(model, weights_path, test_dataloader, device)
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")