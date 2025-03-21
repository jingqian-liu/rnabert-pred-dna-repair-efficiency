import torch
import torch.nn as nn
import torch.optim as optim

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, n_epochs, device='cuda'):
    training_loss = []
    validation_loss = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs1, inputs2, targets in train_loader:
            targets = targets.to(dtype=torch.float32).to(device)
            outputs = model(inputs1, inputs2)
            optimizer.zero_grad()
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_training_loss = epoch_loss / len(train_loader)
        training_loss.append(avg_training_loss)
        print(f'Epoch {epoch+1} training loss: {avg_training_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs1, inputs2, targets in val_loader:
                targets = targets.to(dtype=torch.float32).to(device)
                outputs = model(inputs1, inputs2)
                val_loss += criterion(outputs, targets.unsqueeze(1)).item()

        avg_val_loss = val_loss / len(val_loader)
        validation_loss.append(avg_val_loss)
        print(f'Epoch {epoch+1} validation loss: {avg_val_loss:.4f}')

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    return best_model_state, training_loss, validation_loss
