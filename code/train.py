import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import get_train_dataloader, get_test_dataloader

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

def train(train_loader, test_loader, model, criterion, optimizer, num_epochs):
    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_loader = get_train_dataloader(num_images=512, image_size=(64, 64))
    test_loader = get_test_dataloader(num_images=128, image_size=(64, 64))

    input_size = 64 * 64 * 3
    output_size = 360 # (0-359 degrees)
    model = SimpleModel(input_size, output_size)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 10

    train(train_loader, test_loader, model, criterion, optimizer, num_epochs)

    torch.save(model.state_dict(), "../model_weights/simple_model.pth")