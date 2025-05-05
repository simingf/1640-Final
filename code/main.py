import torch
import torch.nn as nn
import torch.optim as optim
from train import train
from eval import evaluate_model, generate_histogram_vals
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

# train_deg_diff = generate_histogram_vals(model, train_loader, device)
# test_deg_diff = generate_histogram_vals(model, test_loader, device)

# # make separate histograms for train and test using matplotlib
# import matplotlib.pyplot as plt
# import numpy as np

# # Train histogram
# plt.figure(figsize=(10, 5))
# plt.hist(train_deg_diff, bins=50, alpha=0.7, color='blue')
# plt.xlabel('Degree Difference')
# plt.ylabel('Frequency')
# plt.title('Histogram of Train Degree Differences')
# plt.savefig("../train_histogram.png")
# plt.show()

# # Test histogram
# plt.figure(figsize=(10, 5))
# plt.hist(test_deg_diff, bins=50, alpha=0.7, color='orange')
# plt.xlabel('Degree Difference')
# plt.ylabel('Frequency')
# plt.title('Histogram of Test Degree Differences')
# plt.savefig("../test_histogram.png")
# plt.show()