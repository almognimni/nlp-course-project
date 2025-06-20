import torch
import torch.nn as nn
from tqdm import tqdm

from .evaluate import binary_accuracy # Local import

def train_language_model(model, train_loader, val_loader, optimizer, criterion, device, config):
    """Training loop for the language model (Task 1)."""
    train_losses = []
    val_losses = []

    print("\\n--- Starting Language Model Training ---")
    for epoch in range(config.LM_NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.LM_NUM_EPOCHS} [Training]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.fc.out_features), targets.view(-1))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.LM_NUM_EPOCHS} [Validation]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, model.fc.out_features), targets.view(-1))
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{config.LM_NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("--- Finished Training ---")
    return {'train_loss': train_losses, 'val_loss': val_losses}


def train_and_evaluate_classifier(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    """Training loop for the classification models (Task 2)."""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\\n--- Starting Classifier Training ---")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss, epoch_train_acc = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            inputs, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels.float())
            acc = binary_accuracy(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_acc / len(train_loader)

        model.eval()
        epoch_val_loss, epoch_val_acc = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, labels = [t.to(device) for t in batch]
                predictions = model(inputs)
                loss = criterion(predictions, labels.float())
                acc = binary_accuracy(predictions, labels)
                epoch_val_loss += loss.item()
                epoch_val_acc += acc.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = epoch_val_acc / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f'Epoch {epoch+1:02}: | Train Loss: {avg_train_loss:.3f} | Train Acc: {avg_train_acc*100:.2f}% | Val Loss: {avg_val_loss:.3f} | Val Acc: {avg_val_acc*100:.2f}%')

    print("--- Finished Training ---")
    return history