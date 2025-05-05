import torch
import torch.nn as nn
import torch.optim as optim
from train import train
from eval import evaluate_model
from dataloader import get_train_dataloader, get_test_dataloader
from model import ResNetModel

torch.mps.empty_cache()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
train_loader = get_train_dataloader(num_images=8000, image_size=(224, 224))
test_loader = get_test_dataloader(num_images=2000, image_size=(224, 224))

model = ResNetModel().to(device)
model.load_state_dict(torch.load("../model_weights/resnet18.pth"))
# print(model)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)
# num_epochs = 50
# train(train_loader, test_loader, model, criterion, optimizer, num_epochs, device, file_path="../model_weights/resnet18.pth")

train_degree_diff = evaluate_model(model, train_loader, device)
test_degree_diff = evaluate_model(model, test_loader, device)
print(f"Train Average Degree Difference: {train_degree_diff.item():.2f}")
print(f"Test Average Degree Difference: {test_degree_diff.item():.2f}")