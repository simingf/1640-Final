import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import get_train_dataloader, get_test_dataloader
from model import SimpleModel

def train(train_loader, test_loader, model, criterion, optimizer, num_epochs, file_path=None):
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader):
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    if file_path:
        torch.save(model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

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

    train(train_loader, test_loader, model, criterion, optimizer, num_epochs, file_path="../model_weights/simple_model.pth")