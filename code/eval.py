import torch
from dataloader import get_test_dataloader, get_train_dataloader

def evaluate_model(model, weights_path, dataloader, device=None, is_train=False):
        model = model().to(device)
        model.load_state_dict(weights_path)
        model.eval()

        if is_train:
            dataloader = get_train_dataloader(image_size=(64,64))
        else:
            dataloader = get_test_dataloader(image_size=(64,64))
    
        diff = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                labels_degrees = torch.argmax(labels, dim=1)
                diff += torch.abs(preds - labels_degrees)

        accuracy = diff / len(labels)
        return accuracy