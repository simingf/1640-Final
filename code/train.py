import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import get_dataloader

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Hyperparameters
input_size = 64 * 64 * 3
output_size = 360 # (0-359 degrees)
num_epochs = 10
learning_rate = 0.001

model = SimpleModel(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = get_dataloader(num_images=512, image_size=(64, 64))

for epoch in tqdm(range(num_epochs)):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")