import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader  # Assuming get_dataloader is defined in dataloader.py

# Define a simple model with a dense layer
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
input_size = 10  # Adjust based on your dataset
output_size = 1  # Adjust based on your dataset
learning_rate = 0.001
num_epochs = 10

# Initialize model, loss function, and optimizer
model = SimpleModel(input_size, output_size)
criterion = nn.MSELoss()  # Example: Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load data
train_loader = get_dataloader(batch_size=32)  # Adjust batch_size as needed

# Training loop
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

print("Training complete.")